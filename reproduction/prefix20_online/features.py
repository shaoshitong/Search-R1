"""Eight-GPU, frozen layer27 extraction of actual twenty-token policy prefixes.

The question and past observations remain causal encoder context. Only retained
generated positions enter the probe. This module generates no trajectories.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

from reproduction.prefix_only_correctness_ab.features import (
    REVISION, REPRESENTATION, canonical, once, progress, read, remote, sha,
    retained_segments, _LayerCaptured,
)


def scan_rollout(complete_path, output):
    """Convert committed rollout row shards into short causal contexts on CPU."""
    import numpy as np
    complete_path = remote(complete_path)
    complete = read(complete_path)
    if complete['model_revision'] != REVISION:
        raise ValueError('Online rollout must use the fixed official actor')
    if complete['questions'] != 5120 or complete['trajectories'] != 25600:
        raise ValueError('Each fresh epoch requires complete 5120 x G5 trajectories')
    out = remote(output)
    out.mkdir(parents=True, exist_ok=False)
    rows, contexts, policies, positions, offsets = [], [], [], [], [0]
    seen, questions = set(), {}
    for shard in complete['shards']:
        path = remote(shard['path'])
        if path.stat().st_size != shard['bytes'] or sha(path) != shard['sha256']:
            raise ValueError('Committed rollout shard differs: ' + str(path))
        count = 0
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                original = json.loads(line)
                if original.get('epoch', complete['epoch']) != complete['epoch']:
                    raise ValueError('Rollout row belongs to another epoch')
                tid = original['trajectory_id']
                label = original['strict_em']
                if tid in seen or type(label) is not int or label not in (0, 1):
                    raise ValueError('Duplicate trajectory or invalid own-trajectory EM label')
                seen.add(tid)
                qid = str(original['qid'])
                qhash = original['question_hash']
                previous_hash, siblings = questions.get(qid, (qhash, 0))
                if qhash != previous_hash:
                    raise ValueError('Question identity changed within G5')
                questions[qid] = (qhash, siblings + 1)
                if 'prefix_segments' in original:
                    from .model import expected_policy_positions
                    segments = [dict(segment) for segment in original['prefix_segments']]
                    tokens = list(original['policy_token_ids'])
                    pos = list(original['policy_position_ids'])
                    bounds = [{k: segment[k] for k in ('turn_index', 'policy_start', 'policy_length', 'position_start')} for segment in segments]
                    if not 1 <= len(tokens) <= 20 or pos != expected_policy_positions(bounds):
                        raise ValueError('Compact retained-prefix token/position coverage differs')
                    for segment in segments:
                        offset, n = segment['policy_start'], segment['policy_length']
                        request_length = segment['request_length']
                        context = segment['context_tokens']
                        if (request_length != segment['position_start'] or len(context) != request_length + n or
                            context[request_length:] != tokens[offset:offset + n]):
                            raise ValueError('Compact prefix differs from its actual causal request')
                else:
                    segments, tokens, pos = retained_segments(original['behavior_turns'], tid, limit=20)
                if not tokens:
                    raise ValueError('Actual trajectory has no retained policy token')
                records = []
                for segment in segments:
                    context = segment.pop('context_tokens')
                    segment.update(context_offset=len(contexts), context_length=len(context),
                                   context_sha256=hashlib.sha256(canonical(context)).hexdigest())
                    contexts.extend(context)
                    records.append(segment)
                row = dict(row=len(rows), trajectory_id=tid, split='train', qid=qid,
                           question_hash=qhash, strict_em=label, length=len(tokens),
                           prefix_length=len(tokens), policy_offset=offsets[-1], segments=records,
                           segment_bounds=[{k: s[k] for k in ('turn_index', 'policy_start', 'policy_length', 'position_start')} for s in records],
                           epoch=complete['epoch'], source_rollout_path=str(path),
                           source_rollout_sha256=shard['sha256'],
                           policy_prefix_sha256=hashlib.sha256(canonical(tokens)).hexdigest(),
                           position_ids_sha256=hashlib.sha256(canonical(pos)).hexdigest())
                if 'normalized_question_hash' in original:
                    row['normalized_question_hash'] = original['normalized_question_hash']
                rows.append(row)
                policies.extend(tokens)
                positions.extend(pos)
                offsets.append(len(policies))
                count += 1
        if shard.get('count', count) != count:
            raise ValueError('Rollout shard count differs')
    if len(rows) != 25600 or len(questions) != 5120 or {v[1] for v in questions.values()} != {5}:
        raise ValueError('Epoch does not contain 5120 unique complete G5 questions')
    arrays = {'packed_context_ids.npy': np.asarray(contexts, dtype=np.int64),
              'packed_policy_ids.npy': np.asarray(policies, dtype=np.int64),
              'packed_policy_positions.npy': np.asarray(positions, dtype=np.int64),
              'policy_offsets.npy': np.asarray(offsets, dtype=np.int64)}
    for name, array in arrays.items():
        np.save(out / name, array, allow_pickle=False)
    with (out / 'index.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(canonical(row).decode() + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    proof = dict(schema_version=3, role='online20_actual_policy_prefix_scan', epoch=complete['epoch'],
                 representation=REPRESENTATION, count=len(rows), split_counts={'train': len(rows), 'test': 0},
                 encoder_revision=REVISION, model_dir=complete['model_dir'],
                 model_files_sha256=complete['model_files_sha256'], pad_token_id=complete['pad_token_id'],
                 sources_sha256=sha(complete_path), source_complete_path=str(complete_path),
                 selection_sha256=complete['selection_sha256'], head_max_tokens=20, max_policy_prefix_tokens=20,
                 encoder_global_max_tokens=max(s['context_length'] for r in rows for s in r['segments']),
                 prefix_window='multi_turn_actual_kept_policy_tokens',
                 prefix_source='same_trajectory_turn_order_actual_kept_policy_tokens',
                 train_questions=5120, train_trajectories=25600,
                 train_positive=sum(r['strict_em'] for r in rows),
                 files={name: dict(sha256=sha(out / name), bytes=(out / name).stat().st_size)
                        for name in tuple(arrays) + ('index.jsonl',)})
    once(out / 'scan-complete.json', proof)
    return proof


def encode(inputs, output, model_dir, micro_batch_size, attention_implementation='sdpa'):
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModel
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError('Online extraction requires eight actual visible GPUs and ranks')
    source, out, model_path = remote(inputs), remote(output), remote(model_dir)
    proof = read(source / 'scan-complete.json')
    if proof['encoder_revision'] != REVISION or proof['representation'] != REPRESENTATION:
        raise ValueError('Wrong actor/window for online features')
    if int(proof['head_max_tokens']) != 20 or int(proof['count']) != 25600:
        raise ValueError('Online feature input must be exactly 25600 twenty-token trajectories')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    rows = [json.loads(line) for line in (source / 'index.jsonl').read_text().splitlines()]
    contexts = np.load(source / 'packed_context_ids.npy', allow_pickle=False, mmap_mode='r')
    selected, segments, total = [], [], 0
    for original in rows[rank::world]:
        row = dict(original, feature_shard=rank, feature_offset=total)
        row['segments'] = [dict(segment, target_feature_offset=total + segment['policy_start']) for segment in row['segments']]
        selected.append(row)
        segments.extend(row['segments'])
        total += row['length']
    # Long full request contexts are length-bucketed to reduce padding; row/label
    # identity and feature destination offsets stay fixed through the reordering.
    segments.sort(key=lambda s: s['context_length'])
    model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16,
                                     attn_implementation=attention_implementation, local_files_only=True,
                                     trust_remote_code=False, low_cpu_mem_usage=True)
    model.requires_grad_(False).eval().to('cuda:' + str(local))
    if len(model.layers) != 36 or model.config.hidden_size != 2048:
        raise ValueError('Wrong frozen official encoder shape')
    captured = []
    def hook(_module, _inputs, value):
        captured.append(value[0] if isinstance(value, tuple) else value)
        raise _LayerCaptured()
    handle = model.layers[27].register_forward_hook(hook)
    target = out / ('features-shard-%d.npy' % rank)
    array = np.lib.format.open_memmap(target, mode='w+', dtype=np.uint16, shape=(total, 2048))
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(segments), micro_batch_size):
                batch = segments[start:start + micro_batch_size]
                width = max(s['context_length'] for s in batch)
                tokens = torch.full((len(batch), width), proof['pad_token_id'], dtype=torch.long, device=local)
                mask = torch.zeros_like(tokens)
                positions = torch.zeros_like(tokens)
                for i, segment in enumerate(batch):
                    lo, n = segment['context_offset'], segment['context_length']
                    tokens[i, :n] = torch.from_numpy(np.array(contexts[lo:lo + n], copy=True)).to(local)
                    mask[i, :n] = 1
                    positions[i, :n] = torch.arange(n, device=local)
                captured.clear()
                try:
                    model(input_ids=tokens, attention_mask=mask, position_ids=positions,
                          use_cache=False, output_hidden_states=False, return_dict=True)
                except _LayerCaptured:
                    pass
                if len(captured) != 1:
                    raise RuntimeError('Layer27 post-block capture missing')
                hidden = captured.pop()
                for i, segment in enumerate(batch):
                    h = hidden[i, segment['request_length']:segment['context_length']].detach().cpu().contiguous()
                    lo, n = segment['target_feature_offset'], segment['policy_length']
                    if h.shape != (n, 2048):
                        raise RuntimeError('Policy feature shape differs')
                    array[lo:lo + n] = h.view(torch.int16).numpy().view(np.uint16)
                if start % (micro_batch_size * 100) == 0:
                    progress(out, rank, stage='ONLINE20_FROZEN_ENCODING', completed_segments=start + len(batch), total_segments=len(segments))
        array.flush()
    finally:
        handle.remove()
    index_path = out / ('index-shard-%d.jsonl' % rank)
    with index_path.open('x') as handle:
        for row in selected:
            lo, n = row['feature_offset'], row['length']
            row['feature_sha256'] = hashlib.sha256(array[lo:lo + n].tobytes()).hexdigest()
            handle.write(canonical(row).decode() + '\n')
    del array, model
    torch.cuda.empty_cache()
    shard = dict(rank=rank, count=len(selected), total_tokens=total,
                 elapsed_seconds=time.monotonic() - started,
                 files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in (target, index_path)})
    once(out / ('shard-%d-complete.json' % rank), shard)
    dist.barrier()
    if rank == 0:
        shards = [read(out / ('shard-%d-complete.json' % r)) for r in range(world)]
        combined, files = [], {}
        for shard in shards:
            combined.extend(json.loads(line) for line in (out / ('index-shard-%d.jsonl' % shard['rank'])).read_text().splitlines())
            files.update(shard['files'])
        combined.sort(key=lambda row: row['row'])
        if [row['row'] for row in combined] != list(range(25600)):
            raise ValueError('Eight feature shards have missing or repeated rows')
        with (out / 'index.jsonl').open('x') as handle:
            for row in combined:
                handle.write(canonical(row).decode() + '\n')
        files['index.jsonl'] = dict(sha256=sha(out / 'index.jsonl'), bytes=(out / 'index.jsonl').stat().st_size)
        spec = dict(representation=REPRESENTATION, prefix_length=20, feature_layer=27, boundary='post_block',
                    feature_dim=2048, encoder_revision=REVISION, encoder_model_dir=str(model_path),
                    encoder_files_sha256=proof['model_files_sha256'], feature_dtype='bfloat16',
                    encoder_attention_implementation=attention_implementation,
                    global_max_tokens=20, head_max_tokens=20, pad_token_id=proof['pad_token_id'],
                    prefix_window='multi_turn_actual_kept_policy_tokens',
                    encoder_frozen=True, encoder_optimizer_updates=0, question_hidden_stored=False,
                    environment_hidden_stored=False, executed_decoder_indices=list(range(28)))
        manifest = dict(schema_version=3, representation=REPRESENTATION, count=len(combined),
                        split_counts={'train': len(combined), 'test': 0}, feature_dim=2048,
                        global_max_tokens=20, spec=spec, spec_sha=hashlib.sha256(canonical(spec)).hexdigest(),
                        files=files, shards=shards, inputs_dir=str(source), inputs_scan_sha256=sha(source / 'scan-complete.json'),
                        epoch=proof['epoch'], selection_sha256=proof['selection_sha256'])
        once(out / 'manifest.json', manifest)
        once(out / 'features-complete.json', dict(status='ONLINE20_EIGHT_GPU_FEATURES_COMPLETE',
             epoch=proof['epoch'], manifest_sha256=sha(out / 'manifest.json'), count=len(combined), actor_updates=0))
    dist.barrier()
    dist.destroy_process_group()


class FeatureCache:
    """Read new width20 and immutable historical width80 caches without changing them."""
    def __init__(self, path, expected_manifest_sha=None):
        import numpy as np
        self.path = remote(path)
        manifest_path = self.path / 'manifest.json'
        self.manifest_sha = sha(manifest_path)
        if expected_manifest_sha and self.manifest_sha != expected_manifest_sha:
            raise ValueError('Feature manifest SHA differs')
        self.manifest = read(manifest_path)
        self.spec = self.manifest['spec']
        if (self.manifest['representation'] != REPRESENTATION or self.spec['prefix_length'] != 20 or
            self.spec['encoder_revision'] != REVISION or self.spec['feature_layer'] != 27 or
            self.spec['boundary'] != 'post_block' or self.spec.get('question_hidden_stored') is not False or
            self.spec.get('encoder_frozen') is not True or
            self.spec.get('prefix_window') != 'multi_turn_actual_kept_policy_tokens'):
            raise ValueError('Require official frozen layer27 twenty-policy-token feature cache')
        if self.manifest['spec_sha'] != hashlib.sha256(canonical(self.spec)).hexdigest():
            raise ValueError('Feature encoder specification differs')
        self.max_tokens, self.feature_dim = 20, 2048
        self.index, self.ids_by_split, self.shards = {}, {'train': [], 'test': []}, {}
        if sha(self.path / 'index.jsonl') != self.manifest['files']['index.jsonl']['sha256']:
            raise ValueError('Feature identity index differs')
        for shard in self.manifest['shards']:
            name = 'features-shard-%d.npy' % shard['rank']
            path = self.path / name
            if path.stat().st_size != shard['files'][name]['bytes']:
                raise ValueError('Feature shard is incomplete')
            self.shards[shard['rank']] = np.load(path, allow_pickle=False, mmap_mode='r')
            if (self.shards[shard['rank']].dtype != np.uint16 or
                self.shards[shard['rank']].shape != (shard['total_tokens'], 2048)):
                raise ValueError('Feature shard BF16 layout differs')
        for line in (self.path / 'index.jsonl').read_text().splitlines():
            row = json.loads(line)
            tid = row['trajectory_id']
            if tid in self.index or type(row['strict_em']) is not int or row['strict_em'] not in (0, 1) or not 1 <= row['length'] <= 20:
                raise ValueError('Invalid own-trajectory feature/label row')
            self.index[tid] = row
            self.ids_by_split[row['split']].append(tid)
        self.ids = list(self.index)
        if len(self.ids) != self.manifest['count']:
            raise ValueError('Feature index count differs')
        inputs = Path(self.manifest['inputs_dir'])
        self.input_ids = np.load(inputs / 'packed_policy_ids.npy', allow_pickle=False, mmap_mode='r')
        self.position_ids = np.load(inputs / 'packed_policy_positions.npy', allow_pickle=False, mmap_mode='r')
        self.verified = set()

    def get_batch(self, ids, device='cpu'):
        import numpy as np
        import torch
        rows = [self.index[tid] for tid in ids]
        width = max(row['length'] for row in rows)
        features = torch.zeros((len(rows), width, 2048), dtype=torch.bfloat16)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        positions = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            lo, n = row['feature_offset'], row['length']
            raw = np.array(self.shards[row['feature_shard']][lo:lo + n], copy=True)
            if row['trajectory_id'] not in self.verified:
                if hashlib.sha256(raw.tobytes()).hexdigest() != row['feature_sha256']:
                    raise ValueError('Own-trajectory feature bytes differ')
                self.verified.add(row['trajectory_id'])
            features[i, :n] = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
            mask[i, :n] = True
            offset = row['policy_offset']
            positions[i, :n] = torch.from_numpy(np.array(self.position_ids[offset:offset + n], copy=True))
        return dict(features=features.to(device), attention_mask=mask.to(device),
                    position_ids=positions.to(device), policy_segments=[row['segment_bounds'] for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout-complete')
    parser.add_argument('--inputs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model')
    parser.add_argument('--micro-batch-size', type=int, default=32)
    parser.add_argument('--attention-implementation', choices=('sdpa', 'flash_attention_2'), default='sdpa')
    args = parser.parse_args()
    if args.rollout_complete:
        print(json.dumps(scan_rollout(args.rollout_complete, args.output)), flush=True)
    else:
        if not args.inputs or not args.model or args.micro_batch_size < 1:
            parser.error('Encoding requires --inputs, --model and positive --micro-batch-size')
        encode(args.inputs, args.output, args.model, args.micro_batch_size, args.attention_implementation)


if __name__ == '__main__':
    main()
