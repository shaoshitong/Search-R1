"""Frozen causal initial-request + twenty-policy-token features for replay.

All data and tensors stay on remote storage. Existing committed token scans are
read without decoding or re-tokenizing. An epoch is encoded only when replayed;
the fixed diagnostic is encoded once. Question slots occupy [0, Qmax), and the
twenty policy slots always occupy [Qmax, Qmax + 20), including short prefixes.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

from reproduction.prefix_only_correctness_ab.features import (
    REVISION, REPRESENTATION as POLICY_REPRESENTATION, canonical, once,
    progress, read, remote, sha, retained_segments, _LayerCaptured,
)
from reproduction.prefix20_online.features import scan_rollout as _scan_rollout
from reproduction.prefix20_online.model import expected_policy_positions


REPRESENTATION = 'initial_request_and_retained_policy20_layer27_postblock_v1'
FEATURE_ROLE = 'actual_initial_request_plus_same_trajectory_retained_policy20'
POLICY_TOKENS = 20
FEATURE_DIM = 2048


def initial_question_segment(row):
    """Locate the exact initial request; never substitute a later tool context."""
    segments = row['segments']
    if not segments:
        raise ValueError('Missing original initial-request context')
    first = segments[0]
    if first['turn_index'] != 0 or first['policy_start'] != 0:
        raise ValueError('Initial turn0 request missing from retained scan; recover its original IDs from raw source')
    n = first['request_length']
    if type(n) is not int or n < 1 or first['context_length'] < n:
        raise ValueError('Invalid actual initial-request length')
    if first['position_start'] != n:
        raise ValueError('Initial request and native policy positions differ')
    return dict(context_offset=first['context_offset'], context_length=n,
                question_length=n, turn_index=0, position_start=0)


def retained_twenty(row):
    """Trim an authentic width20/80 scan to the same first twenty kept tokens."""
    n = min(POLICY_TOKENS, row.get('policy_length', row['length']))
    if not 1 <= n <= POLICY_TOKENS:
        raise ValueError('Require at least one actual retained policy token')
    chosen = []
    for original in row['segments']:
        if original['policy_start'] >= n:
            break
        take = min(original['policy_length'], n - original['policy_start'])
        segment = dict(original, policy_length=take,
                       context_length=original['request_length'] + take)
        chosen.append(segment)
    bounds = [{k: s[k] for k in ('turn_index', 'policy_start', 'policy_length', 'position_start')}
              for s in chosen]
    if len(expected_policy_positions(bounds)) != n:
        raise ValueError('Actual twenty-token prefix segments are incomplete')
    return chosen, bounds, n


def _source_scan(inputs):
    source = remote(inputs)
    proof = read(source / 'scan-complete.json')
    if (proof['encoder_revision'] != REVISION or
            proof['representation'] not in (POLICY_REPRESENTATION, REPRESENTATION)):
        raise ValueError('Wrong official frozen token scan')
    index_path = source / 'index.jsonl'
    if sha(index_path) != proof['files']['index.jsonl']['sha256']:
        raise ValueError('Source scan identity index differs')
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    if len(rows) != proof['count']:
        raise ValueError('Source token-scan row count differs')
    arrays = remote(proof.get('source_inputs_dir', str(source)))
    return source, proof, rows, arrays


def question_max_from_rows(rows):
    if not rows:
        raise ValueError('Empty question-length source')
    lengths = [initial_question_segment(row)['question_length'] for row in rows]
    return max(lengths), min(lengths), len(lengths)


def initial_request_length_from_rollout(row):
    """Read turn0 length from actual IDs/metadata; future turns are irrelevant."""
    if 'prefix_segments' in row:
        segments = row['prefix_segments']
        if not segments or segments[0]['turn_index'] != 0:
            raise ValueError('Compact rollout omits actual turn0 request; original raw IDs are required')
        segment = segments[0]
        n = segment['request_length']
        if (type(n) is not int or n < 1 or len(segment['context_tokens']) < n or
                segment['position_start'] != n):
            raise ValueError('Actual compact turn0 request identity differs')
    else:
        turns = row['behavior_turns']
        if not turns or turns[0]['turn_index'] != 0 or turns[0]['trajectory_id'] != row['trajectory_id']:
            raise ValueError('Original turn0 trajectory identity differs')
        n = len(turns[0]['actual_request_token_ids'])
        if not n:
            raise ValueError('Empty actual turn0 request')
    return n


def _rollout_question_lengths(complete_path):
    """Inspect compact actual requests directly, without rebuilding old scans."""
    path = remote(complete_path)
    complete = read(path)
    if (complete['model_revision'] != REVISION or complete['questions'] != 5120 or
            complete['trajectories'] != 25600):
        raise ValueError('Qmax requires committed fixed-official 5120xG5 rollout')
    lengths, ids, groups = [], set(), {}
    for shard in complete['shards']:
        source = remote(shard['path'])
        if source.stat().st_size != shard['bytes'] or sha(source) != shard['sha256']:
            raise ValueError('Committed compact rollout differs during Qmax scan')
        opener = gzip.open if source.suffix == '.gz' else open
        count = 0
        with opener(source, 'rt', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                tid = row['trajectory_id']
                if tid in ids or row.get('epoch', complete['epoch']) != complete['epoch']:
                    raise ValueError('Duplicate or wrong-epoch trajectory during Qmax scan')
                ids.add(tid)
                n = initial_request_length_from_rollout(row)
                qid = str(row['qid'])
                qhash, siblings = groups.get(qid, (row['question_hash'], 0))
                if qhash != row['question_hash']:
                    raise ValueError('Question identity changed during Qmax scan')
                groups[qid] = qhash, siblings + 1
                lengths.append(n)
                count += 1
        if count != shard.get('count', count):
            raise ValueError('Compact shard trajectory count differs during Qmax scan')
    if len(lengths) != 25600 or len(groups) != 5120 or {v[1] for v in groups.values()} != {5}:
        raise ValueError('Qmax rollout is not a complete unique 5120 G5 batch')
    return complete, max(lengths), min(lengths), len(lengths)


def scan_question_max(sources, output, expected_train_epochs=70):
    """CPU-only fixed Qmax over every replay batch and the fixed test scan.

    sources is a list (or a JSON path containing a list) of descriptors with
    inputs (existing scan) or rollout_complete, split and epoch. The E1 original
    combined scan can appear twice with split train/test. Later compact rollouts
    are read directly, so pruned historical derived scans need not be rebuilt.
    No feature encoding or full-trajectory forward occurs.
    """
    if isinstance(sources, (str, Path)):
        sources = read(remote(sources))
    if isinstance(sources, dict):
        if 'result' in sources:
            sources = sources['result']
        if 'replay_sources' in sources:
            sources = sources['replay_sources'] + [sources['test_source']]
        else:
            sources = sources['sources']
    records, train_epochs, maximum, test_seen = [], set(), 0, False
    for descriptor in sources:
        descriptor = dict(descriptor)
        if 'epoch' not in descriptor and 'source_data_epoch' in descriptor:
            descriptor['epoch'] = descriptor['source_data_epoch']
        split = descriptor['split']
        if split not in ('train', 'test'):
            raise ValueError('Explicit train/test split required for Qmax')
        if 'rollout_complete' in descriptor:
            if split != 'train':
                raise ValueError('Fixed diagnostic must reuse its original authentic token scan')
            path = remote(descriptor['rollout_complete'])
            if descriptor.get('sha256') and sha(path) != descriptor['sha256']:
                raise ValueError('Committed rollout SHA differs from frozen replay inventory')
            proof, hi, lo, count = _rollout_question_lengths(path)
            if proof['epoch'] != descriptor['epoch']:
                raise ValueError('Qmax descriptor and committed rollout epoch differ')
            provenance = dict(rollout_complete=str(path), complete_sha256=sha(path),
                              compact_shards=proof['shards'])
        else:
            source, proof, rows, _arrays = _source_scan(descriptor['inputs'])
            selected = [row for row in rows if row['split'] == split]
            expected = 25600 if split == 'train' else 2560
            if len(selected) != expected:
                raise ValueError('Qmax source must contain one complete 5120/512 G5 split')
            hi, lo, count = question_max_from_rows(selected)
            provenance = dict(inputs=str(source), scan_sha256=sha(source / 'scan-complete.json'),
                              index_sha256=proof['files']['index.jsonl']['sha256'])
        if split == 'train':
            epoch = descriptor['epoch']
            if type(epoch) is not int or epoch < 1 or epoch in train_epochs:
                raise ValueError('Each replay epoch must appear exactly once in Qmax scan')
            train_epochs.add(epoch)
        elif test_seen:
            raise ValueError('Fixed test split appears more than once in Qmax scan')
        else:
            test_seen = True
        records.append(dict(split=split, epoch=descriptor.get('epoch'), count=count,
                            min_question_tokens=lo, max_question_tokens=hi, **provenance))
        maximum = max(maximum, hi)
    if not test_seen or (expected_train_epochs is not None and
                        train_epochs != set(range(1, expected_train_epochs + 1))):
        raise ValueError('Qmax requires all requested replay epochs and the fixed test split')
    result = dict(schema_version=1, role='all_replay_batches_fixed_initial_question_length',
                  question_max_tokens=maximum, policy_max_tokens=POLICY_TOKENS,
                  head_max_tokens=maximum + POLICY_TOKENS,
                  train_epochs=sorted(train_epochs), test_generated=False,
                  model_loaded=False, gpu_used=False, sources=records)
    once(remote(output), result)
    return result


def prepare_scan(inputs, output, split, epoch=None):
    """Create a small replay index, referencing immutable original token arrays."""
    source, proof, original_rows, arrays = _source_scan(inputs)
    if split not in ('train', 'test'):
        raise ValueError('An explicit train/test split is required')
    out = remote(output)
    out.mkdir(parents=True, exist_ok=False)
    rows, seen = [], set()
    for original in original_rows:
        if original['split'] != split:
            continue
        tid = original['trajectory_id']
        if tid in seen or type(original['strict_em']) is not int or original['strict_em'] not in (0, 1):
            raise ValueError('Duplicate trajectory or invalid own strictEM label')
        seen.add(tid)
        question = initial_question_segment(original)
        segments, bounds, n = retained_twenty(original)
        row = dict(original, row=len(rows), question_length=question['question_length'],
                   question_context_offset=question['context_offset'],
                   policy_length=n, prefix_length=n, length=n, segments=segments,
                   segment_bounds=bounds, source_scan_row=original['row'])
        if epoch is not None:
            row['replay_source_epoch'] = epoch
        rows.append(row)
    expected = 25600 if split == 'train' else 2560
    if len(rows) != expected:
        raise ValueError('Replay split does not have its complete fixed G5 trajectory count')
    with (out / 'index.jsonl').open('x') as handle:
        for row in rows:
            handle.write(canonical(row).decode() + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    result = dict(schema_version=1, role='question20_replay_actual_token_scan',
                  representation=REPRESENTATION, feature_role=FEATURE_ROLE,
                  count=len(rows), split_counts={s: len(rows) if s == split else 0 for s in ('train', 'test')},
                  encoder_revision=REVISION, model_dir=proof['model_dir'],
                  model_files_sha256=proof['model_files_sha256'], pad_token_id=proof['pad_token_id'],
                  source_inputs_dir=str(arrays), source_scan_path=str(source),
                  source_scan_sha256=sha(source / 'scan-complete.json'),
                  source_index_sha256=proof['files']['index.jsonl']['sha256'],
                  source_files=proof.get('source_files', proof['files']),
                  question_max_observed=max(row['question_length'] for row in rows),
                  policy_max_tokens=POLICY_TOKENS, epoch=epoch,
                  selection_sha256=proof.get('selection_sha256'),
                  files={'index.jsonl': dict(sha256=sha(out / 'index.jsonl'),
                                             bytes=(out / 'index.jsonl').stat().st_size)})
    once(out / 'scan-complete.json', result)
    return result


def scan_rollout(complete_path, output):
    """Use the original authentic rollout scanner, then prepare its train split."""
    out = remote(output)
    source = out.with_name(out.name + '-policy-source')
    _scan_rollout(complete_path, source)
    epoch = read(remote(complete_path))['epoch']
    return prepare_scan(source, out, 'train', epoch=epoch)


def encode(inputs, output, model_dir, micro_batch_size, question_max_tokens,
           attention_implementation='sdpa'):
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModel
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 8 or torch.cuda.device_count() != 8 or micro_batch_size < 1:
        raise ValueError('Replay encoding requires eight visible GPUs/ranks and a positive micro batch')
    source, proof, rows, arrays = _source_scan(inputs)
    out, model_path = remote(output), remote(model_dir)
    if proof['representation'] != REPRESENTATION:
        raise ValueError('Prepare a question20 replay scan before encoding')
    qmax = int(question_max_tokens)
    if qmax < proof['question_max_observed']:
        raise ValueError('Fixed Qmax is shorter than an actual initial request')
    contexts = np.load(arrays / 'packed_context_ids.npy', allow_pickle=False, mmap_mode='r')
    policy_ids = np.load(arrays / 'packed_policy_ids.npy', allow_pickle=False, mmap_mode='r')
    policy_positions = np.load(arrays / 'packed_policy_positions.npy', allow_pickle=False, mmap_mode='r')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    selected, segments, total = [], [], 0
    for original in rows[rank::world]:
        qlen, plen = original['question_length'], original['policy_length']
        row = dict(original, feature_shard=rank, feature_offset=total,
                   question_feature_offset=total, policy_feature_offset=total + qlen,
                   feature_length=qlen + plen)
        row['segments'] = [dict(s, target_feature_offset=total + qlen + s['policy_start'],
                                question_target_offset=total if s['turn_index'] == 0 else None,
                                question_length=qlen if s['turn_index'] == 0 else 0)
                           for s in original['segments']]
        selected.append(row)
        segments.extend(row['segments'])
        total += qlen + plen
    segments.sort(key=lambda s: s['context_length'])
    model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16,
                                     attn_implementation=attention_implementation, local_files_only=True,
                                     trust_remote_code=False, low_cpu_mem_usage=True)
    model.requires_grad_(False).eval().to('cuda:' + str(local))
    if len(model.layers) != 36 or model.config.hidden_size != FEATURE_DIM:
        raise ValueError('Wrong official frozen encoder shape')
    captured = []
    def hook(_module, _inputs, value):
        captured.append(value[0] if isinstance(value, tuple) else value)
        raise _LayerCaptured()
    handle = model.layers[27].register_forward_hook(hook)
    target = out / ('features-shard-%d.npy' % rank)
    array = np.lib.format.open_memmap(target, mode='w+', dtype=np.uint16, shape=(total, FEATURE_DIM))
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, len(segments), micro_batch_size):
                batch = segments[start:start + micro_batch_size]
                width = max(s['context_length'] for s in batch)
                tokens = torch.full((len(batch), width), proof['pad_token_id'], dtype=torch.long, device=local)
                mask, positions = torch.zeros_like(tokens), torch.zeros_like(tokens)
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
                    raise RuntimeError('Actual layer27 post-block capture is missing')
                hidden = captured.pop()
                for i, segment in enumerate(batch):
                    req, n = segment['request_length'], segment['policy_length']
                    h = hidden[i, req:req + n].detach().cpu().contiguous()
                    if h.shape != (n, FEATURE_DIM) or h.dtype != torch.bfloat16:
                        raise RuntimeError('Actual policy representation shape/dtype differs')
                    lo = segment['target_feature_offset']
                    array[lo:lo + n] = h.view(torch.int16).numpy().view(np.uint16)
                    if segment['question_target_offset'] is not None:
                        qlen = segment['question_length']
                        qh = hidden[i, :qlen].detach().cpu().contiguous()
                        if qh.shape != (qlen, FEATURE_DIM) or qlen != req:
                            raise RuntimeError('Initial-request representation differs')
                        qlo = segment['question_target_offset']
                        array[qlo:qlo + qlen] = qh.view(torch.int16).numpy().view(np.uint16)
                if start % (micro_batch_size * 100) == 0:
                    progress(out, rank, stage='QUESTION20_LAYER27_ENCODING',
                             completed_segments=start + len(batch), total_segments=len(segments))
        array.flush()
    finally:
        handle.remove()
    index_path = out / ('index-shard-%d.jsonl' % rank)
    with index_path.open('x') as handle:
        for row in selected:
            lo, n = row['feature_offset'], row['feature_length']
            row['feature_sha256'] = hashlib.sha256(array[lo:lo + n].tobytes()).hexdigest()
            off, plen = row['policy_offset'], row['policy_length']
            expected_pos = expected_policy_positions(row['segment_bounds'])
            if policy_positions[off:off + plen].tolist() != expected_pos:
                raise ValueError('Saved policy positions differ from their actual turn contexts')
            for s in row['segments']:
                c, req, take = s['context_offset'], s['request_length'], s['policy_length']
                p = off + s['policy_start']
                if contexts[c + req:c + req + take].tolist() != policy_ids[p:p + take].tolist():
                    raise ValueError('Saved policy tokens differ from actual causal context')
            handle.write(canonical(row).decode() + '\n')
        handle.flush()
        os.fsync(handle.fileno())
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
        for item in shards:
            combined.extend(json.loads(line) for line in (out / ('index-shard-%d.jsonl' % item['rank'])).read_text().splitlines())
            files.update(item['files'])
        combined.sort(key=lambda row: row['row'])
        if [row['row'] for row in combined] != list(range(proof['count'])):
            raise ValueError('Eight feature shards do not cover the exact replay split')
        with (out / 'index.jsonl').open('x') as handle:
            for row in combined:
                handle.write(canonical(row).decode() + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        files['index.jsonl'] = dict(sha256=sha(out / 'index.jsonl'), bytes=(out / 'index.jsonl').stat().st_size)
        spec = dict(representation=REPRESENTATION, feature_role=FEATURE_ROLE,
                    prefix_length=POLICY_TOKENS, policy_max_tokens=POLICY_TOKENS,
                    question_max_tokens=qmax, head_max_tokens=qmax + POLICY_TOKENS,
                    global_max_tokens=qmax + POLICY_TOKENS, feature_layer=27, boundary='post_block',
                    feature_dim=FEATURE_DIM, encoder_revision=REVISION,
                    encoder_model_dir=str(model_path), encoder_files_sha256=proof['model_files_sha256'],
                    feature_dtype='bfloat16', encoder_attention_implementation=attention_implementation,
                    encoder_frozen=True, encoder_optimizer_updates=0,
                    question_hidden_stored=True, question_source='actual_turn0_request_token_ids',
                    environment_hidden_stored=False, labels_passed_to_encoder=False,
                    policy_window='same_trajectory_multi_turn_first20_kept_policy_tokens',
                    slot_layout='question_rightpad_to_Qmax_then_policy_rightpad_to20',
                    policy_absolute_positions='original_native_request_length_plus_policy_offset',
                    pad_token_id=proof['pad_token_id'], executed_decoder_indices=list(range(28)))
        manifest = dict(schema_version=1, representation=REPRESENTATION, count=len(combined),
                        split_counts=proof['split_counts'], feature_dim=FEATURE_DIM,
                        global_max_tokens=qmax + POLICY_TOKENS, spec=spec,
                        spec_sha=hashlib.sha256(canonical(spec)).hexdigest(), files=files, shards=shards,
                        inputs_dir=str(source), source_inputs_dir=str(arrays),
                        inputs_scan_sha256=sha(source / 'scan-complete.json'),
                        epoch=proof['epoch'], selection_sha256=proof.get('selection_sha256'))
        once(out / 'manifest.json', manifest)
        once(out / 'features-complete.json', dict(status='QUESTION20_EIGHT_GPU_FEATURES_COMPLETE',
             manifest_sha256=sha(out / 'manifest.json'), count=len(combined), actor_updates=0,
             question_max_tokens=qmax, head_max_tokens=qmax + POLICY_TOKENS))
    dist.barrier()
    dist.destroy_process_group()


def assemble_batch(rows, shards, position_ids, question_max_tokens, device='cpu'):
    """Pure cache-to-batch layout, with Qmax padding separate from policy padding."""
    import numpy as np
    import torch
    qmax = question_max_tokens
    width = qmax + POLICY_TOKENS
    features = torch.zeros((len(rows), width, FEATURE_DIM), dtype=torch.bfloat16)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    qmask, pmask = torch.zeros_like(mask), torch.zeros_like(mask)
    positions = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        qlen, plen = row['question_length'], row['policy_length']
        if not 1 <= qlen <= qmax or not 1 <= plen <= POLICY_TOKENS:
            raise ValueError('Actual question/policy length exceeds fixed feature layout')
        lo, n = row['feature_offset'], row['feature_length']
        raw = np.array(shards[row['feature_shard']][lo:lo + n], copy=True)
        if raw.shape != (qlen + plen, FEATURE_DIM):
            raise ValueError('Packed question/policy feature length differs')
        if hashlib.sha256(raw.tobytes()).hexdigest() != row['feature_sha256']:
            raise ValueError('Own-trajectory cached feature bytes differ')
        h = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
        features[i, :qlen], features[i, qmax:qmax + plen] = h[:qlen], h[qlen:]
        mask[i, :qlen], mask[i, qmax:qmax + plen] = True, True
        qmask[i, :qlen], pmask[i, qmax:qmax + plen] = True, True
        positions[i, :qlen] = torch.arange(qlen)
        off = row['policy_offset']
        pos = np.array(position_ids[off:off + plen], copy=True)
        if pos.tolist() != expected_policy_positions(row['segment_bounds']):
            raise ValueError('Original policy RoPE positions differ')
        positions[i, qmax:qmax + plen] = torch.from_numpy(pos)
    return dict(features=features.to(device), attention_mask=mask.to(device),
                position_ids=positions.to(device), question_mask=qmask.to(device), policy_mask=pmask.to(device),
                question_lengths=torch.tensor([r['question_length'] for r in rows], dtype=torch.long, device=device),
                policy_lengths=torch.tensor([r['policy_length'] for r in rows], dtype=torch.long, device=device),
                policy_segments=[r['segment_bounds'] for r in rows], feature_role=FEATURE_ROLE)


class FeatureCache:
    def __init__(self, path, expected_manifest_sha=None, source_inputs_override=None):
        import numpy as np
        self.path = remote(path)
        self.manifest_sha = sha(self.path / 'manifest.json')
        if expected_manifest_sha and self.manifest_sha != expected_manifest_sha:
            raise ValueError('Feature manifest SHA differs')
        self.manifest, self.spec = read(self.path / 'manifest.json'), None
        self.spec = self.manifest['spec']
        if (self.manifest['representation'] != REPRESENTATION or
                self.spec['feature_role'] != FEATURE_ROLE or self.spec['feature_layer'] != 27 or
                self.spec['boundary'] != 'post_block' or self.spec['encoder_revision'] != REVISION or
                self.spec['question_hidden_stored'] is not True or not self.spec['encoder_frozen'] or
                self.spec['prefix_length'] != POLICY_TOKENS or
                self.manifest['spec_sha'] != hashlib.sha256(canonical(self.spec)).hexdigest()):
            raise ValueError('Require actual initial-request + policy20 frozen layer27 features')
        self.question_max_tokens = self.spec['question_max_tokens']
        self.max_tokens, self.feature_dim = self.question_max_tokens + POLICY_TOKENS, FEATURE_DIM
        if self.spec['head_max_tokens'] != self.max_tokens or self.manifest['global_max_tokens'] != self.max_tokens:
            raise ValueError('Fixed flatten head width differs')
        self.index, self.ids_by_split, self.shards = {}, {'train': [], 'test': []}, {}
        if sha(self.path / 'index.jsonl') != self.manifest['files']['index.jsonl']['sha256']:
            raise ValueError('Feature identity index differs')
        for shard in self.manifest['shards']:
            name = 'features-shard-%d.npy' % shard['rank']
            p = self.path / name
            if p.stat().st_size != shard['files'][name]['bytes']:
                raise ValueError('Feature shard is incomplete')
            array = np.load(p, allow_pickle=False, mmap_mode='r')
            if array.dtype != np.uint16 or array.shape != (shard['total_tokens'], FEATURE_DIM):
                raise ValueError('Feature shard BF16 packed layout differs')
            self.shards[shard['rank']] = array
        for line in (self.path / 'index.jsonl').read_text().splitlines():
            row = json.loads(line)
            tid = row['trajectory_id']
            if tid in self.index or type(row['strict_em']) is not int or row['strict_em'] not in (0, 1):
                raise ValueError('Invalid own-trajectory feature identity/label')
            if row['feature_length'] != row['question_length'] + row['policy_length']:
                raise ValueError('Question and policy packed feature coverage differs')
            self.index[tid] = row
            self.ids_by_split[row['split']].append(tid)
        self.ids = list(self.index)
        if len(self.ids) != self.manifest['count'] or {k: len(v) for k, v in self.ids_by_split.items()} != self.manifest['split_counts']:
            raise ValueError('Incomplete feature cache split')
        arrays = remote(source_inputs_override if source_inputs_override is not None
                        else self.manifest['source_inputs_dir'])
        self.source_inputs_original = self.manifest['source_inputs_dir']
        self.source_inputs_actual = str(arrays)
        self.position_ids = np.load(arrays / 'packed_policy_positions.npy', allow_pickle=False, mmap_mode='r')
        self.input_ids = np.load(arrays / 'packed_policy_ids.npy', allow_pickle=False, mmap_mode='r')

    def get_batch(self, ids, device='cpu'):
        return assemble_batch([self.index[tid] for tid in ids], self.shards, self.position_ids,
                              self.question_max_tokens, device)

    def load(self, row_indices, device='cpu'):
        """Optional index-based adapter, with labels derived from each own trajectory."""
        import torch
        ids = [self.ids[i] for i in row_indices]
        batch = self.get_batch(ids, device)
        labels = torch.tensor([self.index[tid]['strict_em'] for tid in ids], dtype=torch.float32, device=device)
        return batch['features'], batch['attention_mask'], batch['position_ids'], labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question-max-sources')
    parser.add_argument('--expected-train-epochs', type=int, default=70)
    parser.add_argument('--rollout-complete')
    parser.add_argument('--prepare-inputs')
    parser.add_argument('--split', choices=('train', 'test'))
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--inputs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model')
    parser.add_argument('--question-max-tokens', type=int)
    parser.add_argument('--micro-batch-size', type=int, default=32)
    parser.add_argument('--attention-implementation', choices=('sdpa', 'flash_attention_2'), default='sdpa')
    args = parser.parse_args()
    if args.question_max_sources:
        print(json.dumps(scan_question_max(args.question_max_sources, args.output, args.expected_train_epochs)), flush=True)
    elif args.rollout_complete:
        print(json.dumps(scan_rollout(args.rollout_complete, args.output)), flush=True)
    elif args.prepare_inputs:
        if not args.split:
            parser.error('Preparing a split requires --split')
        print(json.dumps(prepare_scan(args.prepare_inputs, args.output, args.split, args.epoch)), flush=True)
    else:
        if not args.inputs or not args.model or not args.question_max_tokens:
            parser.error('Encoding requires --inputs --model --question-max-tokens')
        encode(args.inputs, args.output, args.model, args.micro_batch_size,
               args.question_max_tokens, args.attention_implementation)


if __name__ == '__main__':
    main()
