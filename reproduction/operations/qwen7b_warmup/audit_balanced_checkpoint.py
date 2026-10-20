"""Independent CPU audit of immutable balanced Prefix20 recovery states.

This does not run training or accept the five-epoch experiment as complete.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from .download_model import atomic, fingerprints


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def accept(base, receipt=None):
    base = Path(base)
    config = json.loads((base / 'config.json').read_text())
    publication = json.loads((base / 'preparation/source-publication.json').read_text())
    receipt = receipt or json.loads((base / 'probe/latest.json').read_text())
    assert config['prefix_tokens'] == 20 and config['fit_world'] == 4
    assert config['source_checkpoint']['steps'] == 10464
    assert receipt['source_commit'] == config['source_commit'] == publication['commit']
    assert receipt['source_steps'] == 10464 and receipt['actor_updates'] == 0
    assert 2 <= receipt['epoch'] <= config['fit_epochs']
    assert 0 <= receipt['cursor'] <= 1318
    expected = 10464 + 1318 * (receipt['epoch'] - 2)
    expected += 1318 if receipt['completed_epoch'] else receipt['cursor']
    assert receipt['steps'] == expected and receipt['steps'] > 10464
    assert receipt['completed_epoch'] == (receipt['cursor'] == 0)
    audit = base / 'balanced-checkpoint-audit'
    audit.mkdir(exist_ok=True)
    # The pre-evaluation state and sealed epoch have the same optimizer step,
    # but distinct cursor/metrics/RNG metadata and file hashes. Keep both.
    path = audit / ('epoch-%04d-step-%08d-cursor-%04d-%s.pt' %
                    (receipt['epoch'], receipt['steps'], receipt['cursor'], receipt['sha256'][:12]))
    if not path.exists():
        os.link(receipt['path'], path)
    actual = fingerprints(path)
    assert actual['sha256'] == receipt['sha256'] and actual['bytes'] == receipt['bytes']
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert state['role'] == 'EPOCH1_BALANCED_RANK1_CONTINUATION'
    for key in ('epoch', 'cursor', 'completed_epoch', 'steps', 'source_steps', 'actor_updates', 'source_commit'):
        assert state[key] == receipt[key], key
    assert state['source_checkpoint'] == config['source_checkpoint']
    assert state['config_sha256'] == fingerprints(base / 'config.json')['sha256']
    assert state['bank_sha256'] == config['mixed_bank_sha256'] == fingerprints(base / 'bank/groups.json')['sha256']
    fit = Path(publication['code_root']) / 'reproduction/operations/qwen7b_warmup/balanced_finetune.py'
    assert state['fit_code_sha256'] == config['fit_file_sha256'] == fingerprints(fit)['sha256']
    from .balanced_sampling import plan,proof
    groups=json.loads((base/'bank/groups.json').read_text())['train']
    sampling=proof(groups,plan(groups,config['seed'],state['epoch'],config['fit_epoch_groups']))
    assert state['sampling_plan']==sampling
    assert sampling['mixed']==sampling['nonmixed']==33599 and sampling['updates']==1318
    ranks = state['rank_states']
    assert [rank['rank'] for rank in ranks] == list(range(4))
    hashes = {}
    for key, record in [('raw', 'model_sha256'), ('ema', 'ema_sha256')]:
        own = state[key]
        assert tuple(own['block.self_attn.q_proj.weight'].shape) == (3584, 3584)
        assert tuple(own['block.self_attn.k_proj.weight'].shape) == (512, 3584)
        assert tuple(own['block.mlp.gate_proj.weight'].shape) == (18944, 3584)
        assert tuple(own['output.weight'].shape) == (1, (273 + 20) * 3584)
        assert all(bool(torch.isfinite(value).all()) for value in own.values())
        hashes[key] = state_hash(own)
        assert all(rank[record] == hashes[key] for rank in ranks)
    for key in ('raw_buffers', 'ema_buffers'):
        assert isinstance(state[key], dict)
        assert all(bool(torch.isfinite(value).all()) for value in state[key].values())
    optimizer = state['optimizer']
    digest = hashlib.sha256()
    steps, parameter_ids = set(), []
    ordered_parameters=list(state['raw'].values());parameter_index=0
    for group in optimizer['param_groups']:
        assert group['lr'] == 5e-6 and group['weight_decay'] == 0
        for parameter in group['params']:
            parameter_ids.append(parameter)
            item = optimizer['state'][parameter]
            assert set(item) == {'step', 'exp_avg', 'exp_avg_sq'}
            steps.add(int(item['step']))
            assert item['exp_avg'].shape == item['exp_avg_sq'].shape == ordered_parameters[parameter_index].shape
            parameter_index+=1
            for key, value in sorted(item.items()):
                assert bool(torch.isfinite(value).all())
                digest.update(key.encode())
                digest.update(value.contiguous().numpy().tobytes())
    assert len(parameter_ids) == len(set(parameter_ids)) == len(optimizer['state']) == len(state['raw'])
    assert steps == {state['steps']}
    hashes['adam'] = digest.hexdigest()
    assert all(rank['optimizer_sha256'] == hashes['adam'] and rank['steps'] == state['steps'] for rank in ranks)
    for rank in ranks:
        rng = rank['rng']
        assert set(rng) == {'python', 'numpy', 'torch', 'cuda'}
        assert rng['torch'].numel() > 0 and rng['cuda'].numel() > 0
    result = dict(status='BALANCED_PREFIX20_INDEPENDENT_CPU_SHA_RAW_EMA_ADAM_FOUR_RANK_RNG_VERIFIED',
                  immutable_checkpoint=str(path), checkpoint=receipt, hashes=hashes,
                  checked_ranks=4, optimizer_parameter_tensors=len(parameter_ids),
                  pending_evaluation=not receipt['completed_epoch'] and receipt['cursor'] == 1318,
                  next_update_proof=str(base / 'probe/next-update-proof.json'),
                  audit_source_sha256=fingerprints(__file__)['sha256'], time=time.time(),
                  experiment_completion_claimed=False)
    atomic(audit / ('step-%08d.json' % state['steps']), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    receipt = json.loads(args.receipt.read_text()) if args.receipt else None
    print(json.dumps(accept(args.base, receipt)), flush=True)


if __name__ == '__main__':
    main()
