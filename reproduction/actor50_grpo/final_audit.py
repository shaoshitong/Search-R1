"""Independent remote CPU acceptance after owned training/evaluation has exited.

No forward pass, rollout, optimizer write, cleanup, or existing-file mutation.
Only the dedicated audit directory receives progress/proof/error records.
"""
import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

RUN = '54ac33dcebe64e62a11fe4177f41f116'
EXECUTION = '21bf70af3e9b4cabbab7862e9532069c'
RUNTIME = '2d9712395f6c666cd0b53f843f8bc1bfb6e5d11d'
SOURCES = {
    '2wikimultihopqa': 12576, 'bamboogle': 125, 'hotpotqa': 7405,
    'musique': 2417, 'nq': 3610, 'popqa': 14267, 'triviaqa': 11313}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value: raise ValueError('Duplicate JSON key: ' + key)
        value[key] = item
    return value


def parse(raw):
    def bad(value): raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(raw, object_pairs_hook=unique_object, parse_constant=bad)
    canonical(value)  # Also rejects overflow such as1e999.
    return value


def read(path):
    return parse(Path(path).read_bytes())


def identity(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def regular(path):
    p = Path(path)
    assert p.is_absolute() and str(p).startswith('/data/') and p.resolve() == p
    assert not p.is_symlink() and p.is_file()
    return p


class Audit:
    def __init__(self, base, out):
        self.base, self.out = Path(base), Path(out)
        self.hashes, self.payloads = {}, []

    def progress(self, phase, **fields):
        p = self.out/'progress.json'
        t = p.with_suffix('.partial')
        t.write_bytes(canonical(dict(phase=phase, checked_unix=time.time(), **fields))+b'\n')
        os.replace(t, p)

    def seal(self, path, expected=None):
        p = regular(path); before = identity(p)
        if before not in self.hashes:
            h = hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
            assert identity(p) == before
            self.hashes[before] = h.hexdigest()
        row = dict(path=str(p), bytes=before[2], sha256=self.hashes[before],
                   dev=before[0], inode=before[1], mtime_ns=before[3], independent_full_sha=True)
        if expected:
            assert row['sha256'] == expected['sha256'] and row['bytes'] == expected['bytes']
        self.payloads.append(row)
        return row

    def document(self, path, expected_sha=None):
        seal = self.seal(path)
        if expected_sha: assert seal['sha256'] == expected_sha
        value = read(path)
        assert identity(path) == (seal['dev'], seal['inode'], seal['bytes'], seal['mtime_ns'])
        return value, dict(seal, canonical_finite_sha256=digest(value))

    def torch_cpu(self, path, expected):
        import torch
        seal = self.seal(path, expected)
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert identity(path) == (seal['dev'], seal['inode'], seal['bytes'], seal['mtime_ns'])
        return saved

    def ledger(self, actor, outer):
        attempt = Path(actor['hf_model_dir']).parent
        results = []
        for rank in range(4):
            p = regular(attempt/('rank-%d-updates.jsonl'%rank)); before = identity(p)
            with p.open('rb') as f:
                offset = max(0, before[2]-2*1024**2); f.seek(offset)
                lines = f.read().splitlines()
            assert identity(p) == before
            if offset: lines = lines[1:]
            rows = [parse(line) for line in lines]
            writes = [x for x in rows if x.get('event') == 'ACTUAL_ADAM_WRITE']
            opps = [x for x in rows if x.get('event') == 'OPPORTUNITY']
            expected = 18 if outer == 88 else 20
            assert len(opps) == 20 and len({x['opportunity'] for x in opps}) == 20
            assert len(writes) == expected and len({x['opportunity'] for x in writes}) == expected
            assert all(x['rank'] == rank and x['outer'] == outer for x in writes+opps)
            assert writes[-1]['actual_adam_steps'] == actor['actual_adam_steps_per_rank'][rank]
            results.append(dict(rank=rank, actual_returns=len(writes), opportunities=20,
                                tail_sha256=hashlib.sha256(b'\n'.join(lines)).hexdigest()))
        return results

    def checkpoint(self, joint, outer, config):
        import torch
        from safetensors import safe_open
        from reproduction.prefix_selection200.checkpoint_cpu_once import tensor_cpu, validate_adam, validate_rng
        from reproduction.prefix_selection200.actor import learning_rate
        from reproduction.prefix_selection200.candidate_schedule import consumed_candidates_through
        a, p = joint['actor'], joint['probe']; steps = 20*outer-(2 if outer >= 88 else 0)
        self.progress('CHECKPOINT_ACTOR', outer=outer)
        ranks = []
        for rank, item in enumerate(a['rank_states']):
            self.progress('CHECKPOINT_ACTOR_RANK_SHA_FINITE', outer=outer, rank=rank)
            saved = self.torch_cpu(item['path'], item)
            assert saved['run_id'] == RUN and saved['rank'] == rank and saved['completed_outer'] == outer
            assert saved['actual_adam_steps'] == steps and saved['binding'] == a['binding']
            assert saved['grounding_state'] == joint['grounding']
            assert saved['optimizer_layout'] == 'FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world4'
            shapes = [v['shape'] for group in saved['optimizer_parameter_layout'] for v in group]
            count = validate_adam(saved['optimizer'], shapes, steps, learning_rate(outer), weight_decay=.01)
            assert saved['scheduler']['last_epoch'] == outer
            assert saved['scheduler']['_step_count'] == outer+1
            assert saved['scheduler']['_last_lr'] == [learning_rate(outer)]
            ranks.append(dict(rank=rank, Adam=steps, scheduler=outer, states=count, rng=validate_rng(saved['rng'])))
            del saved; gc.collect()
        tensor_count = 0
        for name, item in a['hf_files'].items():
            path = Path(a['hf_model_dir'])/name; self.seal(path, item)
            if name.endswith('.safetensors'):
                self.progress('CHECKPOINT_HF_SHA_FINITE', outer=outer, file=name)
                with safe_open(str(path), framework='pt', device='cpu') as f:
                    for key in f.keys():
                        tensor_cpu(f.get_tensor(key), dtype=torch.float32, finite=True); tensor_count += 1
        saved = self.torch_cpu(a['driver_state']['path'], a['driver_state'])
        assert saved['completed_outer'] == outer and saved['actor_version'] == a['actor_version']
        assert saved['grounding_state'] == joint['grounding']
        pool, pool_seal = self.document(joint['pool_snapshot'], joint['pool_sha256'])
        assert saved['candidate_pool_state']['value'] == pool
        assert pool['next_outer'] == outer+1
        assert pool['cycle']*len(pool['order'])+pool['cursor'] == pool['consumed_candidates']
        assert sorted(pool['order']) == list(range(len(pool['order'])))
        assert pool['consumed_candidates'] == consumed_candidates_through(config, outer)
        driver_rng = validate_rng(saved['rng']); del saved; gc.collect()
        self.progress('CHECKPOINT_PROBE_SHA_FINITE', outer=outer)
        saved = self.torch_cpu(joint['probe_snapshot'], p)
        assert saved['run_id'] == RUN and saved['completed_outer'] == p['completed_outer']
        assert saved['actual_adam_steps_per_rank'] == p['actual_adam_steps_per_rank']
        assert saved['binding_sha256'] == p['binding_sha256'] == digest(saved['binding'])
        for v in saved['model'].values(): tensor_cpu(v, dtype=torch.float32, finite=True)
        for v in saved.get('model_nonpersistent_buffers', {}).values(): tensor_cpu(v, finite=True)
        probe_steps = p['actual_adam_steps_per_rank'][0]
        if probe_steps:
            validate_adam(saved['optimizer'], [v.shape for v in saved['model'].values()], probe_steps,
                          saved['learning_rate'], weight_decay=0.)
        else: assert not saved['optimizer']['state']
        assert len(saved['rank_rng']) == 4
        for rank, rng in enumerate(saved['rank_rng']):
            assert rng['rank'] == rank and rng['actual_adam_steps'] == probe_steps
            validate_rng(rng)
        probe_binding = saved['binding']; del saved; gc.collect()
        if outer >= 100:
            assert probe_binding['prefix_tokens'] == 40 and probe_binding['head_max_tokens'] == 313
        ground_snapshot, ground_seal = self.document(joint['grounding_snapshot'], joint['grounding_snapshot_sha256'])
        assert ground_snapshot == joint['grounding']
        return dict(outer=outer, ranks=ranks, hf_finite_tensors=tensor_count, grounding_snapshot=ground_seal,
                    driver_rng=driver_rng, pool=pool_seal, probe_Adam=probe_steps,
                    probe_completed_outer=p['completed_outer'])

    def evaluation(self, config):
        from transformers import AutoTokenizer
        from reproduction.prefix_selection200.eval import load_questions, score_tokens, CONTRACT
        from reproduction.prefix_selection200.interaction_protocol import CONTRACT as SHARED
        self.progress('FINAL_QA_IDENTITIES_AND_SCORES')
        b = self.base; final, final_seal = self.document(b/'final-evaluation/complete.json')
        assert final['run_id'] == RUN and final['actor_version'] == RUN+'-actor0200'
        assert final['qa_count'] == 51713 and final['actual_compute_dtype'] == 'torch.bfloat16'
        assert final['evaluation_contract'] == CONTRACT and final['interaction_protocol'] == SHARED
        assert set(final['actual_worker_exit_codes'].values()) == {0} and len(final['actual_worker_exit_codes']) == 4
        assert self.seal(config['final_questions'])['sha256'] == final['question_source_sha256'] == config['final_questions_sha256']
        questions = load_questions(config['final_questions'], 51713)
        assert Counter(x['data_source'] for x in questions) == SOURCES
        tokenizer = AutoTokenizer.from_pretrained(config['official_model_dir'], local_files_only=True, trust_remote_code=False)
        rows = []
        for rank, meta in enumerate(final['rank_receipts']):
            item, seal = self.document(meta['path'], meta['sha256'])
            assert item['rank'] == rank and item['run_id'] == RUN
            assert item['actor_version'] == final['actor_version'] and item['evaluation_contract'] == CONTRACT
            assert item['interaction_protocol'] == SHARED and item['actual_compute_dtype'] == 'torch.bfloat16'
            assert item['question_source_sha256'] == final['question_source_sha256']
            expected_ordinals = list(range(rank, 51713, 4))
            assert [x['evaluation_ordinal'] for x in item['rows']] == expected_ordinals
            assert item['qa_count'] == len(expected_ordinals)
            full = item['full_output']; self.seal(full['path'], full)
            by_ordinal = {x['evaluation_ordinal']: x for x in item['rows']}; observed = set()
            with gzip.open(full['path'], 'rt', encoding='utf8') as stream:
                for line in stream:
                    native = parse(line); ordinal = native['evaluation_ordinal']
                    assert ordinal in by_ordinal and ordinal not in observed
                    observed.add(ordinal); row = by_ordinal[ordinal]; q = questions[ordinal]
                    assert row['qid'] == native['qid'] == q['qid']
                    assert row['data_source'] == native['data_source'] == q['data_source']
                    assert type(row['em']) is int and row['em'] in (0, 1)
                    assert type(row['f1']) in (int, float) and math.isfinite(row['f1']) and 0 <= row['f1'] <= 1
                    measured = score_tokens(tokenizer, native['initial_prompt_token_ids'], native['response_token_ids'],
                        native['response_info_mask'], q['reward_model']['ground_truth']['target'])
                    for key in ('em', 'answer_present', 'answer_sha256'): assert measured[key] == row[key] == native[key]
                    assert abs(measured['f1']-row['f1']) < 1e-12 and native['f1'] == row['f1']
                    if len(observed) % 512 == 0: self.progress('FINAL_QA_STRICT_RESCORE', rank=rank, verified=len(observed))
            assert observed == set(expected_ordinals); rows.extend(item['rows'])
        rows.sort(key=lambda x: x['evaluation_ordinal'])
        assert [x['evaluation_ordinal'] for x in rows] == list(range(51713))
        expected_ids = [x['qid'] for x in questions]; actual_ids = [x['qid'] for x in rows]
        assert actual_ids == expected_ids and len(set(actual_ids)) == 51713
        assert Counter(x['data_source'] for x in rows) == SOURCES
        assert sum(x['em'] for x in rows) == 24588
        assert final['em'] == sum(x['em'] for x in rows)/51713
        assert abs(final['f1']-sum(x['f1'] for x in rows)/51713) < 1e-12
        for source, count in SOURCES.items():
            subset = [x for x in rows if x['data_source'] == source]; m = final['per_source'][source]
            assert m['qa_count'] == count and m['em'] == sum(x['em'] for x in subset)/count
            assert abs(m['f1']-sum(x['f1'] for x in subset)/count) < 1e-12
        return dict(complete_metadata=final_seal, question_count=51713, strict_rescored=51713,
                    ordered_ids_sha256=digest(actual_ids), expected_ordered_ids_sha256=digest(expected_ids),
                    sources=SOURCES, em=final['em'], f1=final['f1'], per_source=final['per_source'])

    def run(self):
        import torch
        torch.set_num_threads(2); assert not torch.cuda.is_initialized()
        from reproduction.operations.a800_i1_i6_boundary_handoff import owned_present
        from reproduction.actor50_grpo.bootstrap import validate_config
        b = self.base; config = read(b/'preparation/launch-config.json'); validate_config(config, b)
        assert config['run_id'] == RUN and config['execution_id'] == EXECUTION and config['source_commit'] == RUNTIME
        assert read(b/'logs/exit.json')['success'] and read(b/'logs/cleanup-status.json')['owned_births_all_gone']
        assert not owned_present(b)
        completion, completion_seal = self.document(b/'logs/pipeline-complete.json')
        assert completion['run_id'] == RUN and completion['complete_outers'] == 200
        handle = read(b/'preparation/deployment.json')
        for name, expected in handle['source_files'].items(): assert self.seal(b/'repo'/name)['sha256'] == expected
        chain, checkpoints, grounding_proofs = [], [], {}
        for outer in range(1, 201):
            self.progress('FULL200_CHAIN', outer=outer)
            j, seal = self.document(b/'outers'/('outer-%06d'%outer)/'complete.json')
            a = j['actor']; steps = 20*outer-(2 if outer >= 88 else 0)
            assert j['run_id'] == RUN and j['outer'] == a['completed_outer'] == a['scheduler_completed_outer'] == outer
            assert a['actual_adam_steps_per_rank'] == [steps]*4
            ledgers = self.ledger(a, outer)
            ground = j['grounding']
            if ground['version'] not in grounding_proofs:
                from reproduction.prefix_selection200.grounding import checked_state
                checked_state(config, ground, require_adopted=True)
                manifest, manifest_seal = self.document(ground['manifest_path'], ground['manifest_sha256'])
                adopted, adoption_seal = self.document(ground['adoption_receipt'], ground['adoption_receipt_sha256'])
                assert adopted['adopted_unix'] <= j['observed_unix']
                import joblib
                import numpy as np
                heads = []
                for key, name in (('boundary', 'boundary_probe.pkl'), ('commit', 'commit_probe.pkl')):
                    head_seal = self.seal(ground[key]); assert head_seal['sha256'] == ground['head_sha256'][name]
                    fitted = joblib.load(ground[key])
                    assert list(fitted['clf'].classes_) == [0, 1] and np.isfinite(fitted['auc'])
                    from reproduction.a800.recovery_binding import finite_fitted_values
                    finite_fitted_values(vars(fitted['clf']))
                    heads.append(head_seal)
                grounding_proofs[ground['version']] = dict(round=ground['round'], manifest=manifest_seal,
                    adoption=adoption_seal, heads=heads, adopted_before_first_used_outer=outer)
            if outer > 50:
                assert ground['round'] == min(((outer-1)//50)*50, 150)
            if outer > 50:
                assert j['source_lineage']['actor50_joint_sha256'] == config['source_joint50_sha256']
                assert j['source_lineage']['source_warmup_sha256'] == config['probe_source_sha256']
                expected_probe = (min(outer, 190)//10-5)*50
                assert j['probe']['actual_adam_steps_per_rank'] == [expected_probe]*4
            if outer == 50: assert seal['sha256'] == config['source_joint50_sha256']
            if outer in (5, 50, 100, 150, 200): checkpoints.append(self.checkpoint(j, outer, config))
            if outer == 200: assert completion['latest_joint'] == j
            chain.append(dict(outer=outer, metadata_sha256=seal['sha256'], canonical_finite_sha256=seal['canonical_finite_sha256'],
                              actor_Adam=steps, ledgers=ledgers))
            del j; gc.collect()
        evaluation = self.evaluation(config)
        assert completion['final_evaluation'] == read(b/'final-evaluation/complete.json')
        assert not owned_present(b) and not torch.cuda.is_initialized()
        return dict(status='INDEPENDENT_FULL200_CHECKPOINTS_FINAL51713_STRICT_CPU_ACCEPTANCE_PASS',
                    run_id=RUN, execution_id=EXECUTION, runtime_commit=RUNTIME, checked_unix=time.time(),
                    chain=chain, chain_sha256=digest(chain), retained_joint_checkpoints=checkpoints,
                    pipeline_complete=completion_seal, evaluation=evaluation, payload_seals=self.payloads,
                    grounding_proofs=grounding_proofs,
                    owned_writers_gone=True, CUDA_initialized=False, no_new_rollout_or_forward=True,
                    no_existing_training_or_evaluation_file_modified=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); base, out = Path(args.base), Path(args.output)
    assert base.name.endswith(EXECUTION) and out.parent == base and not out.exists()
    out.mkdir(); audit = Audit(base, out)
    (out/'intent.json').write_bytes(canonical(dict(run_id=RUN, execution_id=EXECUTION, checked_unix=time.time()))+b'\n')
    try:
        result = audit.run(); (out/'complete.json').write_bytes(canonical(result)+b'\n')
        summary = {k: v for k, v in result.items() if k not in ('chain', 'payload_seals', 'retained_joint_checkpoints', 'grounding_proofs')}
        summary['chain_count'] = len(result['chain']); summary['milestones'] = [x['outer'] for x in result['retained_joint_checkpoints']]
        summary['complete_sha256'] = hashlib.sha256((out/'complete.json').read_bytes()).hexdigest()
        (out/'summary.json').write_bytes(canonical(summary)+b'\n')
        print(json.dumps(summary, allow_nan=False))
    except BaseException as error:
        (out/'error.json').write_bytes(canonical(dict(error=repr(error), traceback=traceback.format_exc(), checked_unix=time.time()))+b'\n')
        raise


if __name__ == '__main__': main()
