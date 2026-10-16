"""Fixed official-encoder prefix scoring and whole-G5 online probe updates.

The scoring scan has no outcome labels. Only selected official-encoder features
are retained remotely; their own EM labels are joined after the same prefixes
have actually completed. Warmup CP100 supplies model weights only. Online Adam
starts at zero and continues for ten real synchronized writes per actor outer.
"""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from reproduction.prefix20_group_joint100.inline_encoder import InlineEncoder, normalized_row
from reproduction.prefix20_group_joint100.loss import joint_loss
from reproduction.prefix20_group_joint100.train_epoch import (
    batch_to_device, ddp_mean_scale, evaluate_predictions, rank_question_slice,
)
from reproduction.prefix20_online.features import REVISION, canonical, once, read, remote, sha
from reproduction.prefix20_online.launch import publish
from reproduction.prefix20_online.model import expected_policy_positions, validate_prefix_positions
from reproduction.prefix20_online.train_epoch import append, optimizer_learning_rates
from reproduction.prefix20_question_replay.features import FeatureCache, _source_scan
from reproduction.prefix20_question_replay.model import QuestionPrefixCorrectnessProbe, masked_flatten
from reproduction.prefix_only_correctness_ab.train_ddp import actual_adam_steps


SOURCE_SHA = '8df6bd445945ffbbf6a94171f24934495f0ce3ff17a3d2577bbd53d4b8fefd56'
SOURCE_RUN = 'cee20bcd24194b79a9a8a88809bd5886'
TEST_SHA = 'a125fc5af90e34fcd728c82077d9d62070a78d5ef703ac11f3748687d088e64f'
QMAX, WIDTH, DIM, LR = 273, 293, 2048, 1e-6
TENSOR_FIELDS = ('features', 'attention_mask', 'position_ids', 'question_mask',
                 'policy_mask', 'question_lengths', 'policy_lengths')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def online_plan(seed, outer):
    """Exactly512 questions, ten writes, eight equal natural rank workloads."""
    if type(outer) is not int or not 1 <= outer <= 200:
        raise ValueError('Online probe outer must be1..200')
    shuffled = list(range(512))
    random.Random(seed + outer).shuffle(shuffled)
    plan, offset, extras_start = [], 0, 0
    for step in range(10):
        n = 52 if (step + 1) % 5 == 0 else 51
        counts = [6] * 8
        for j in range(n - 48):
            counts[(extras_start + j) % 8] += 1
        plan.append(dict(question_indices=shuffled[offset:offset + n], global_questions=n,
                         rank_questions=counts, extras_start=extras_start))
        offset += n
        extras_start = (extras_start + n - 48) % 8
    assert offset == 512 and all(sum(row['rank_questions'][r] for row in plan) == 64 for r in range(8))
    return plan


def summarize_loss_curve(curve):
    if len(curve) != 10 or [row['step'] for row in curve] != list(range(1, 11)):
        raise ValueError('A complete online outer has ten actual global loss observations')
    return dict(points=curve, first=curve[0], last=curve[-1],
                delta_end_minus_start={key: curve[-1][key]-curve[0][key]
                                      for key in ('trajectory_bce', 'group_bce', 'total_loss')},
                loss_semantics='each_step_global_true_group_weighted_mean_before_its_own_update',
                batches_differ_between_steps=True, monotonic_decrease_required=False,
                actual_adam_updates_per_rank=10, world_size=8, replica_updates_are_not_summed=True)


def candidate_groups(index, expected_questions=5120):
    """Group by authentic question identity, never by consecutive five rows."""
    groups = {}
    for ordinal, (tid, row) in enumerate(index.items()):
        qid = str(row['qid'])
        if qid not in groups:
            groups[qid] = dict(qid=qid, question_hash=row['question_hash'],
                               candidate_ordinal=row.get('candidate_ordinal', len(groups)),
                               trajectory_ids=[], trajectory_ordinals=[])
        group = groups[qid]
        if (row['question_hash'] != group['question_hash'] or
                row.get('candidate_ordinal', group['candidate_ordinal']) != group['candidate_ordinal']):
            raise ValueError('One question changed its authentic candidate identity')
        group['trajectory_ids'].append(tid)
        group['trajectory_ordinals'].append(row.get('trajectory_ordinal', row.get('response_index', ordinal)))
    result = list(groups.values())
    if len(result) != expected_questions or len(index) != expected_questions * 5:
        raise ValueError('Require exactly the requested complete G5 question count')
    ordinals = [row['candidate_ordinal'] for row in result]
    if len(set(ordinals)) != len(ordinals) or any(type(x) is not int or x < 0 for x in ordinals):
        raise ValueError('Candidate ordinals must be unique nonnegative integers')
    if len({row['question_hash'] for row in result}) != len(result):
        raise ValueError('Candidate questions cannot duplicate normalized question content')
    for group in result:
        if len(group['trajectory_ids']) != 5 or len(set(group['trajectory_ids'])) != 5:
            raise ValueError('Incomplete or duplicate authentic G5 trajectory identities')
        order = sorted(range(5), key=lambda j: group['trajectory_ordinals'][j])
        group['trajectory_ids'] = [group['trajectory_ids'][j] for j in order]
        del group['trajectory_ordinals']
    return sorted(result, key=lambda row: row['candidate_ordinal'])


def mixed_probabilities(logits):
    """Same differentiable32-pattern event mass as training, without labels."""
    if logits.ndim != 2 or logits.shape[1] != 5 or not logits.is_floating_point():
        raise ValueError('Expected own-G5 floating correctness logits')
    z = logits if logits.dtype == torch.float64 else logits.float()
    numbers = torch.arange(32, device=z.device).unsqueeze(1)
    shifts = torch.arange(5, device=z.device).unsqueeze(0)
    bits = ((numbers >> shifts) & 1).bool()
    log_patterns = torch.where(bits.unsqueeze(0), torch.nn.functional.logsigmoid(z).unsqueeze(1),
                               torch.nn.functional.logsigmoid(-z).unsqueeze(1)).sum(-1)
    mixed = torch.logsumexp(log_patterns[:, 1:31], dim=1)
    extreme = torch.logsumexp(log_patterns[:, (0, 31)], dim=1)
    return torch.sigmoid(mixed - extreme)


def stable_top512(groups, logits_by_id, k=512):
    if len(groups) < k or set(logits_by_id) != {tid for row in groups for tid in row['trajectory_ids']}:
        raise ValueError('Score every authentic candidate trajectory exactly once')
    logits = torch.tensor([[logits_by_id[tid] for tid in row['trajectory_ids']] for row in groups], dtype=torch.float64)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Actual nonfinite correctness logits cannot rank candidates')
    qs = mixed_probabilities(logits).tolist()
    ps = torch.sigmoid(logits).tolist()
    rows = [dict(group, q=float(q), trajectory_logits=logits[j].tolist(),
                 trajectory_probabilities=ps[j]) for j, (group, q) in enumerate(zip(groups, qs))]
    return sorted(rows, key=lambda row: (-row['q'], row['candidate_ordinal']))[:k], rows


def join_own_em(index, records, actor_version, prefix_scan_sha, selection_sha):
    """Outcome labels must refer to these selected, actually continued prefixes."""
    labels = {}
    for row in records:
        tid = row['trajectory_id']
        if tid in labels or tid not in index or type(row['strict_em']) is not int or row['strict_em'] not in (0, 1):
            raise ValueError('Duplicate, unselected, or invalid own EM outcome')
        if str(row['qid']) != str(index[tid]['qid']) or row['actor_version'] != actor_version:
            raise ValueError('Own outcome belongs to another question or actor version')
        if (row.get('prefix_scan_sha256') != prefix_scan_sha and row.get('selection_sha256') != selection_sha):
            raise ValueError('Own outcome does not bind the scored/continued prefix set')
        expected_prefix = index[tid].get('prefix_sha256')
        if expected_prefix is not None and row.get('prefix_sha256') != expected_prefix:
            raise ValueError('Own outcome replaced the scored trajectory prefix')
        labels[tid] = row['strict_em']
    if set(labels) != set(index):
        raise ValueError('Require exactly all2560 selected own trajectory outcomes')
    return {tid: dict(row, strict_em=labels[tid]) for tid, row in index.items()}


def prefix_row(original):
    """A real initial EOS may leave no kept policy tokens; keep its question."""
    if original.get('length', original.get('prefix_length')) != 0:
        return normalized_row(original, QMAX)
    initial = original.get('initial_context', {})
    length = initial.get('request_length', original.get('question_length'))
    offset = initial.get('context_offset', original.get('question_context_offset'))
    if (type(length) is not int or not 1 <= length <= QMAX or type(offset) is not int or offset < 0
            or original.get('segments') or original.get('segment_bounds')):
        raise ValueError('Actual zero-policy EOS requires its authentic initial prompt context')
    return dict(original, question_length=length, question_context_offset=offset,
                length=0, policy_length=0, prefix_length=0, segments=[], segment_bounds=[])


def validate_online_layout(batch, max_positions):
    features, mask, positions = batch['features'], batch['attention_mask'], batch['position_ids']
    lengths, question, policy = batch['question_lengths'], batch['question_mask'], batch['policy_mask']
    if (features.ndim != 3 or features.shape[1:] != (WIDTH, DIM) or mask.shape != features.shape[:2]
            or positions.shape != mask.shape or positions.dtype != torch.long or mask.dtype != torch.bool
            or question.shape != mask.shape or policy.shape != mask.shape
            or question.dtype != torch.bool or policy.dtype != torch.bool
            or lengths.shape != (features.shape[0],) or lengths.dtype != torch.long):
        raise ValueError('Question+20 head requires the fixed authentic feature layout')
    slots = torch.arange(WIDTH, device=mask.device).unsqueeze(0)
    if (bool(((lengths < 1) | (lengths > QMAX)).any()) or not torch.equal(question, slots < lengths[:, None])
            or bool(policy[:, :QMAX].any()) or not torch.equal(mask, question | policy)
            or not torch.equal(positions[question], slots.expand_as(mask)[question])):
        raise ValueError('Question slots, masks, or original RoPE positions differ')
    for i, bounds in enumerate(batch['policy_segments']):
        if bool(policy[i, QMAX:].any()):
            validate_prefix_positions(policy[i:i+1, QMAX:], positions[i:i+1, QMAX:], [bounds], max_positions)
        elif bounds:
            raise ValueError('Actual zero-token prefix cannot have synthetic policy segments')


class OnlineQuestionProbe(QuestionPrefixCorrectnessProbe):
    """Identical learned tensors; permit authentic initial-EOS question-only rows."""
    def forward(self, batch):
        validate_online_layout(batch, self.max_positions)
        features, mask, positions = batch['features'].detach(), batch['attention_mask'], batch['position_ids']
        kwargs = dict(attention_mask=mask, position_ids=positions,
                      position_embeddings=self.rotary(features, positions), use_cache=False, output_attentions=False)
        if self._cache_position:
            kwargs['cache_position'] = torch.arange(WIDTH, device=features.device)
        hidden = self.block(features, **kwargs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.output(masked_flatten(hidden, mask)).squeeze(-1).float()


class PrefixEncoder(InlineEncoder):
    """The official frozen encoder consumes actual tokens, never outcome labels."""
    def set_inputs(self, inputs_dir):
        import numpy as np
        source, proof, rows, arrays = _source_scan(inputs_dir)
        if any('strict_em' in row for row in rows):
            raise ValueError('Pre-outcome prefix scoring scan must not expose final EM labels')
        self.path, self.manifest, self.manifest_sha = source, proof, sha(source / 'scan-complete.json')
        # The policy actor evolves, while this encoder remains the original
        # official model. Their file maps must never be conflated.
        self.spec = dict(encoder_revision=REVISION, encoder_files_sha256=proof['encoder_files_sha256'])
        self.index = {row['trajectory_id']: prefix_row(row) for row in rows}
        if len(self.index) != len(rows):
            raise ValueError('Duplicate own trajectory identity in prefix scan')
        self.contexts = np.load(arrays / 'packed_context_ids.npy', mmap_mode='r', allow_pickle=False)
        self.input_ids = np.load(arrays / 'packed_policy_ids.npy', mmap_mode='r', allow_pickle=False)
        self.position_ids = np.load(arrays / 'packed_policy_positions.npy', mmap_mode='r', allow_pickle=False)
        self.actual_encoder_batches, self.oom_fallbacks = [], []

    def get_batch(self, ids, device=None):
        import numpy as np
        device = self.device
        rows = [self.index[tid] for tid in ids]
        n = len(rows)
        features = torch.zeros((n, WIDTH, DIM), dtype=torch.bfloat16, device=device)
        question = torch.zeros((n, WIDTH), dtype=torch.bool, device=device)
        policy, positions = torch.zeros_like(question), torch.zeros((n, WIDTH), dtype=torch.long, device=device)
        work = []
        for i, row in enumerate(rows):
            qlen, plen = row['question_length'], row['policy_length']
            question[i, :qlen] = True
            policy[i, QMAX:QMAX+plen] = True
            positions[i, :qlen] = torch.arange(qlen, device=device)
            offset = row.get('policy_offset', 0)
            actual = np.asarray(self.position_ids[offset:offset+plen])
            expected = expected_policy_positions(row['segment_bounds']) if plen else []
            if actual.tolist() != expected:
                raise ValueError('Policy positions differ from their authentic causal segments')
            positions[i, QMAX:QMAX+plen] = torch.as_tensor(actual.copy(), device=device)
            segments = row['segments'] or [dict(context_offset=row['question_context_offset'],
                context_length=qlen, request_length=qlen, policy_start=0, policy_length=0, turn_index=0)]
            for segment in segments:
                lo, req, take = segment['context_offset'], segment['request_length'], segment['policy_length']
                poff = offset + segment['policy_start']
                if not np.array_equal(self.contexts[lo+req:lo+req+take], self.input_ids[poff:poff+take]):
                    raise ValueError('Own prefix IDs do not match the actual causal request')
                work.append(dict(segment, batch_row=i, question_length=qlen))
        work.sort(key=lambda row: row['context_length'])
        cursor, micro = 0, self.micro_batch_size
        while cursor < len(work):
            selected = work[cursor:cursor+micro]
            width = max(row['context_length'] for row in selected)
            tokens = torch.zeros((len(selected), width), dtype=torch.long, device=device)
            valid, rope = torch.zeros_like(tokens), torch.zeros_like(tokens)
            for j, segment in enumerate(selected):
                lo, length = segment['context_offset'], segment['context_length']
                actual = np.asarray(self.contexts[lo:lo+length])
                if len(actual) != length:
                    raise ValueError('Actual prefix context is incomplete')
                tokens[j, :length] = torch.as_tensor(actual.copy(), device=device)
                valid[j, :length] = 1
                rope[j, :length] = torch.arange(length, device=device)
            try:
                hidden = self._capture(tokens, valid, rope).detach()
            except torch.cuda.OutOfMemoryError:
                del tokens, valid, rope
                torch.cuda.empty_cache()
                if micro == 1:
                    raise
                self.oom_fallbacks.append(dict(previous=micro, actual_oom=True, next=max(1, micro//2)))
                micro = max(1, micro//2)
                continue
            if hidden.shape != (len(selected), width, DIM):
                raise ValueError('Frozen official layer27 post-block shape differs')
            for j, segment in enumerate(selected):
                i, req, plen = segment['batch_row'], segment['request_length'], segment['policy_length']
                start = QMAX + segment['policy_start']
                features[i, start:start+plen] = hidden[j, req:req+plen].to(torch.bfloat16)
                if segment['turn_index'] == 0:
                    qlen = segment['question_length']
                    if qlen != req:
                        raise ValueError('Actual question and turn0 request differ')
                    features[i, :qlen] = hidden[j, :qlen].to(torch.bfloat16)
            self.actual_encoder_batches.append(len(selected))
            cursor += len(selected)
            del tokens, valid, rope, hidden
        return dict(features=features.detach(), attention_mask=question | policy, position_ids=positions,
                    question_mask=question, policy_mask=policy,
                    question_lengths=torch.tensor([row['question_length'] for row in rows], device=device),
                    policy_lengths=torch.tensor([row['policy_length'] for row in rows], device=device),
                    policy_segments=[row['segment_bounds'] for row in rows], trajectory_ids=list(ids),
                    qids=[str(row['qid']) for row in rows])


def batch_subset(batch, positions):
    return {key: value[positions].contiguous() if isinstance(value, torch.Tensor)
            else [value[i] for i in positions] if isinstance(value, list) else value
            for key, value in batch.items()}


def concatenate(batches):
    if not batches:
        return None
    return {key: torch.cat([batch[key] for batch in batches]) if isinstance(batches[0][key], torch.Tensor)
            else [item for batch in batches for item in batch[key]]
            for key in batches[0]}


def inference_logits(model, batch, initial_micro, fallbacks):
    """Inference-only OOM reduction; never change an online gradient batch."""
    length = len(batch['trajectory_ids'])
    result, offset, micro = [], 0, min(initial_micro, length)
    while offset < length:
        selected = batch_subset(batch, list(range(offset, min(length, offset+micro))))
        try:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                value = model(selected)
        except torch.cuda.OutOfMemoryError:
            del selected
            torch.cuda.empty_cache()
            if micro == 1:
                raise
            fallbacks.append(dict(previous=micro, actual_oom=True, next=max(1, micro//2)))
            micro = max(1, micro//2)
            continue
        result.append(value.detach().cpu())
        offset += len(selected['trajectory_ids'])
    return torch.cat(result)


class SelectedFeatures:
    def __init__(self, directory):
        self.path = remote(directory)
        self.manifest = read(self.path / 'manifest.json')
        self.manifest_sha = sha(self.path / 'manifest.json')
        if (self.manifest['role'] != 'selected_actual_prefix_official_encoder_features'
                or self.manifest['count'] != 2560 or self.manifest['question_max_tokens'] != QMAX
                or self.manifest['head_max_tokens'] != WIDTH):
            raise ValueError('Selected own-prefix feature manifest differs')
        rows = read(self.path / 'index.json')
        if sha(self.path / 'index.json') != self.manifest['index_sha256']:
            raise ValueError('Selected feature index differs from its producer receipt')
        self.index = {row['trajectory_id']: row for row in rows}
        if len(self.index) != 2560:
            raise ValueError('Selected features must contain2560 unique own trajectories')
        self.shards = {row['rank']: row for row in self.manifest['shards']}
        self.loaded = {}

    def get_batch(self, ids, device):
        single = []
        for tid in ids:
            row = self.index[tid]
            rank = row['feature_rank']
            if rank not in self.loaded:
                path = self.path / self.shards[rank]['file']
                if path.stat().st_size != self.shards[rank]['bytes']:
                    raise ValueError('Selected feature shard size differs')
                self.loaded[rank] = torch.load(path, map_location='cpu', weights_only=False)
            batch = self.loaded[rank]
            i = row['feature_row']
            if batch['trajectory_ids'][i] != tid:
                raise ValueError('Selected feature row no longer belongs to its own trajectory')
            single.append(batch_subset(batch, [i]))
        return batch_to_device(concatenate(single), device)


def atomic_checkpoint(directory, saved, completed_outer):
    directory = remote(directory)
    temporary, path = directory / 'latest.pt.partial', directory / 'latest.pt'
    if temporary.exists():
        raise FileExistsError('Uncommitted online probe checkpoint requires inspection')
    torch.save(saved, temporary)
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    receipt = dict(run_id=saved['run_id'], completed_outer=completed_outer,
                   actual_adam_steps_per_rank=saved['actual_adam_steps_per_rank'],
                   learning_rate=LR, sha256=sha(path), bytes=path.stat().st_size,
                   binding_sha256=saved['binding_sha256'], observed_utc=utc())
    publish(directory / 'latest.json', receipt)
    if completed_outer in (0, 5, 50, 100, 150, 200):
        os.link(path, directory / ('head-outer-%04d.pt' % completed_outer))
    return receipt


def initial_online_state(source, binding):
    """Deliberately import only learned weights, no warmup optimizer/RNG clock."""
    if (source['run_id'] != SOURCE_RUN or source['epoch'] != 100
            or source['binding']['question_max_tokens'] != QMAX or source['binding']['head_max_tokens'] != WIDTH
            or source['test_manifest_sha256'] != TEST_SHA or len(source['model']) != 14
            or source['model']['output.weight'].shape != (1, WIDTH*DIM)
            or any(value.dtype != torch.float32 for value in source['model'].values())):
        raise ValueError('Warmup100 model weights or official question20 architecture differs')
    return dict(schema_version=1, run_id=binding['run_id'], binding=binding, binding_sha256=digest(binding),
                completed_outer=0, model=source['model'], optimizer=None, rank_rng=None,
                actual_adam_steps_per_rank=[0]*8, learning_rate=LR)


def initialize(args):
    directory, source = remote(args.state_dir), remote(args.source_checkpoint)
    if (directory / 'initialization-complete.json').exists():
        existing = read(directory / 'initialization-complete.json')
        if existing['run_id'] != args.run_id or existing['source_sha256'] != args.source_sha256:
            raise ValueError('Existing online probe belongs to another initialization')
        return dict(status='EXISTING_PROBE_INITIALIZATION_NO_REPLAY', receipt=existing)
    if args.source_sha256 != SOURCE_SHA or args.test_manifest_sha != TEST_SHA:
        raise ValueError('This formal run starts from the approved warmup100/fixed diagnostic')
    directory.mkdir(parents=True, exist_ok=True)
    once(directory / 'initialization-intent.json', dict(run_id=args.run_id, source=str(source), source_sha256=SOURCE_SHA))
    if sha(source) != SOURCE_SHA:
        raise ValueError('Approved warmup CP100 bytes differ')
    saved = torch.load(source, map_location='cpu', weights_only=False)
    test = FeatureCache(args.test_cache, args.test_manifest_sha)
    binding = dict(role='prefix_selection200_online_correctness_probe', run_id=args.run_id,
        encoder_revision=REVISION, encoder_files_sha256=test.spec['encoder_files_sha256'],
        question_max_tokens=QMAX, head_max_tokens=WIDTH, feature_layer=27, feature_boundary='post_block',
        prefix_tokens=20, source_probe_sha256=SOURCE_SHA, warmup_model_weights_only=True,
        warmup_optimizer_imported=False, online_initial_adam_steps=0, learning_rate=LR,
        group_weight=1.0, loss='mean_trajectory_BCE_plus_mean_G5_mixed_BCE',
        updates_per_outer=10, global_batch_average=256, actual_global_batches=[255, 260],
        world_size=8, accumulation=1, rank_questions_per_outer=64, rank_trajectories_per_outer=320,
        diagnostic_manifest_sha256=TEST_SHA, diagnostic_questions=512,
        selected_questions=512, candidate_questions=5120, exploration=0,
        selection='q_descending_then_original_candidate_ordinal', zero_policy_eos='authentic_question_only_no_fake_token')
    initial = initial_online_state(saved, binding)
    receipt = atomic_checkpoint(directory, initial, 0)
    result = dict(status='WARMUP100_MODEL_ONLY_ONLINE_ADAM0_INITIALIZED', run_id=args.run_id,
        source_sha256=SOURCE_SHA, source_model_tensors=14, warmup_optimizer_imported=False,
        warmup_rng_imported=False, warmup_updates_reexecuted=0, receipt=receipt)
    once(directory / 'initialization-complete.json', result)
    return result


def setup_ddp():
    rank, local, world = (int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 8 or torch.cuda.device_count() != 8:
        raise ValueError('Probe phases require the actual eight CUDA ranks')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    return rank, local, torch.device('cuda', local)


def load_online(args, device):
    directory = remote(args.state_dir)
    receipt = read(directory / 'latest.json')
    if receipt['completed_outer'] != args.outer - 1 or receipt['actual_adam_steps_per_rank'] != [10*(args.outer-1)]*8:
        raise ValueError('Probe phase requires the immediately preceding complete online boundary')
    path = directory / 'latest.pt'
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        if before.st_size != receipt['bytes']:
            raise ValueError('Online probe checkpoint size differs from its committed receipt')
        saved = torch.load(handle, map_location='cpu', weights_only=False)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Online checkpoint changed while loading its committed boundary')
    if (saved['binding_sha256'] != digest(saved['binding']) or saved['binding_sha256'] != receipt['binding_sha256']
            or saved['completed_outer'] != args.outer-1 or saved['actual_adam_steps_per_rank'] != receipt['actual_adam_steps_per_rank']
            or saved['binding']['learning_rate'] != LR):
        raise ValueError('Online probe binding or actual Adam boundary differs')
    model = OnlineQuestionProbe(remote(args.model_dir), QMAX).to(device=device, dtype=torch.float32)
    model.load_state_dict(saved['model'], strict=True)
    return model, saved, receipt


def score(args):
    rank, local, device = setup_ddp()
    encoder = None
    try:
        model, saved, receipt = load_online(args, device)
        encoder = PrefixEncoder(args.prefix_inputs, args.model_dir, question_max_tokens=QMAX,
                                micro_batch_size=args.encoder_micro_batch_size, device=device)
        manifest = encoder.manifest
        actor_version = manifest['actor_version']
        if (manifest['outer'] != args.outer or manifest['run_id'] != saved['run_id']
                or encoder.spec['encoder_files_sha256'] != saved['binding']['encoder_files_sha256']):
            raise ValueError('Candidate prefix scan belongs to another outer/run/frozen encoder')
        if any(row.get('actor_version') != actor_version for row in encoder.index.values()):
            raise ValueError('A candidate prefix belongs to a different frozen actor version')
        groups = candidate_groups(encoder.index)
        own_groups = groups[rank::8]
        out = remote(args.output)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        local_logits, cached, head_fallbacks = [], [], []
        group_batch = max(1, args.encoder_micro_batch_size // 5)
        model.eval()
        started = time.monotonic()
        with torch.no_grad():
            for offset in range(0, len(own_groups), group_batch):
                ids = [tid for group in own_groups[offset:offset+group_batch] for tid in group['trajectory_ids']]
                batch = encoder.get_batch(ids, device)
                # Scoring has no outcome labels and never calls a supervised loss.
                logits = inference_logits(model, batch, args.encoder_micro_batch_size, head_fallbacks)
                local_logits.extend(zip(ids, logits.cpu().tolist()))
                cached.append(batch_to_device(batch, 'cpu'))
                publish(out / ('score-progress-rank-%d.json' % rank),
                        dict(outer=args.outer, scored_trajectories=len(local_logits), stage='OFFICIAL_ENCODER_PREFIX_SCORING'))
        gathered = [None]*8
        dist.all_gather_object(gathered, local_logits)
        merged = [pair for rows in gathered for pair in rows]
        if len(merged) != 25600 or len({tid for tid, _ in merged}) != 25600:
            raise ValueError('All eight scoring ranks must produce25600 unique own-prefix scores')
        selected, all_scores = stable_top512(groups, dict(merged))
        selection = dict(schema_version=1, run_id=saved['run_id'], outer=args.outer,
            actor_version=actor_version, prefix_scan_sha256=encoder.manifest_sha,
            probe_before_checkpoint_sha256=receipt['sha256'], selected=selected,
            candidate_questions=5120, selected_questions=512, trajectories_per_question=5,
            exploration=0, tie_break='original_candidate_ordinal_ascending',
            formula='1-prod(p_j)-prod(1-p_j)', outcome_labels_used=False)
        selection_sha = digest(selection)
        wanted = {tid for row in selected for tid in row['trajectory_ids']}
        kept = []
        for batch in cached:
            positions = [j for j, tid in enumerate(batch['trajectory_ids']) if tid in wanted]
            if positions:
                kept.append(batch_subset(batch, positions))
        del cached
        local_features = concatenate(kept)
        file = out / ('selected-features-rank-%d.pt' % rank)
        torch.save(local_features, file)
        shard = dict(rank=rank, file=file.name, count=0 if local_features is None else len(local_features['trajectory_ids']),
                     sha256=sha(file), bytes=file.stat().st_size,
                     actual_encoder_batches=encoder.actual_encoder_batches, actual_oom_fallbacks=encoder.oom_fallbacks,
                     actual_head_inference_oom_fallbacks=head_fallbacks)
        local_index = [dict(encoder.index[tid], feature_rank=rank, feature_row=j)
                       for j, tid in enumerate(local_features['trajectory_ids'] if local_features else [])]
        gathered_shards, gathered_index = [None]*8, [None]*8
        dist.all_gather_object(gathered_shards, shard)
        dist.all_gather_object(gathered_index, local_index)
        if rank == 0:
            once(out / 'selection.json', selection)
            actual_selection_sha = sha(out / 'selection.json')
            with (out / 'candidate-scores.jsonl').open('x') as handle:
                for row in all_scores:
                    handle.write(canonical(row).decode()+'\n')
            index = [row for rows in gathered_index for row in rows]
            if len(index) != 2560 or {row['trajectory_id'] for row in index} != wanted:
                raise ValueError('Saved features must retain exactly the selected own prefixes')
            once(out / 'index.json', index)
            result = dict(schema_version=1, role='selected_actual_prefix_official_encoder_features',
                run_id=saved['run_id'], outer=args.outer, count=2560, question_max_tokens=QMAX, head_max_tokens=WIDTH,
                encoder_revision=REVISION, encoder_files_sha256=encoder.spec['encoder_files_sha256'],
                actor_version=actor_version, prefix_scan_sha256=encoder.manifest_sha,
                probe_before_checkpoint_sha256=receipt['sha256'], selection_sha256=actual_selection_sha,
                selection_canonical_sha256=selection_sha, index_sha256=sha(out / 'index.json'),
                candidate_scores_sha256=sha(out / 'candidate-scores.jsonl'),
                shards=gathered_shards, no_outcome_labels=True, encoder_forward='official_frozen_layer27_postblock',
                selected_features_reused_for_online_training=True, elapsed_seconds=time.monotonic()-started)
            once(out / 'manifest.json', result)
            once(out / 'complete.json', dict(status='PREFIX5120G5_STABLE_TOP512_SCORE_COMPLETE', outer=args.outer,
                run_id=saved['run_id'], manifest_sha256=sha(out / 'manifest.json'), selection_sha256=actual_selection_sha,
                selected_questions=512, selected_trajectories=2560, actor_version=actor_version, observed_utc=utc()))
        dist.barrier()
    finally:
        if encoder is not None:
            encoder.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def update(args):
    import numpy as np
    rank, local, device = setup_ddp()
    try:
        model, saved, receipt = load_online(args, device)
        features = SelectedFeatures(args.scored_features)
        m = features.manifest
        if (m['run_id'] != saved['run_id'] or m['outer'] != args.outer
                or m['probe_before_checkpoint_sha256'] != receipt['sha256']
                or m['encoder_files_sha256'] != saved['binding']['encoder_files_sha256']):
            raise ValueError('Online update is not bound to its own scored prefix/features/probe')
        own_path = remote(args.own_em)
        if args.own_em_sha256 and sha(own_path) != args.own_em_sha256:
            raise ValueError('Own-EM result differs from its completion receipt')
        records = [json.loads(line) for line in own_path.read_text().splitlines()]
        index = join_own_em(features.index, records, m['actor_version'], m['prefix_scan_sha256'], m['selection_sha256'])
        groups = candidate_groups(index, 512)
        test = FeatureCache(args.test_cache, args.test_manifest_sha)
        if args.test_manifest_sha != TEST_SHA or test.spec['encoder_files_sha256'] != m['encoder_files_sha256']:
            raise ValueError('Fixed diagnostic or official encoder identity changed')
        if ({str(row['qid']) for row in index.values()} & {str(row['qid']) for row in test.index.values()}
                or {row['question_hash'] for row in index.values()} & {row['question_hash'] for row in test.index.values()}):
            raise ValueError('Selected training questions overlap the fixed diagnostic')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
        expected_before = 10*(args.outer-1)
        if args.outer > 1:
            optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if isinstance(value, torch.Tensor) and key != 'step':
                        state[key] = value.to(device)
            rng = saved['rank_rng'][rank]
            random.setstate(rng['python']); np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch_cpu']); torch.cuda.set_rng_state(rng['torch_cuda'], device)
        else:
            random.seed(args.seed+rank); np.random.seed(args.seed+rank)
            torch.manual_seed(args.seed+rank); torch.cuda.manual_seed_all(args.seed+rank)
        if actual_adam_steps(optimizer) != expected_before or any(rate != LR for rate in optimizer_learning_rates(optimizer)):
            raise ValueError('Online Adam state or constant LR changed')
        plan = online_plan(args.seed, args.outer)
        binding_values = [None]*8
        dist.all_gather_object(binding_values, (features.manifest_sha, sha(own_path), digest(plan), receipt['sha256']))
        if len(set(binding_values)) != 1:
            raise ValueError('Eight ranks differ in selected features, outcomes, plan, or prior probe')
        out = remote(args.output)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        model = DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
        model.train()
        totals = [0.0, 0.0, 0.0, 0, 0]
        global_curve = []
        started = time.monotonic()
        for step, planned in enumerate(plan, start=1):
            local_groups = [groups[j] for j in rank_question_slice(planned, rank)]
            ids = [tid for group in local_groups for tid in group['trajectory_ids']]
            batch = features.get_batch(ids, device)
            labels = torch.tensor([[index[tid]['strict_em'] for tid in group['trajectory_ids']]
                                   for group in local_groups], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(batch).reshape(len(local_groups), 5)
            losses = joint_loss(logits.float(), labels, group_weight=1.0)
            backward_loss = losses['loss'] * ddp_mean_scale(len(local_groups), planned['global_questions'])
            finite = torch.isfinite(backward_loss).to(dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite.item()):
                raise FloatingPointError('Actual nonfinite online probe loss')
            backward_loss.backward()
            finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all().int()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite.item()):
                raise FloatingPointError('Actual nonfinite online probe gradient')
            optimizer.step()
            actual = actual_adam_steps(optimizer)
            if actual != expected_before+step:
                raise RuntimeError('Actual online probe Adam counter differs from its real write budget')
            n = len(local_groups)
            totals[0] += float(losses['trajectory_bce'].detach())*n
            totals[1] += float(losses['group_bce'].detach())*n
            totals[2] += float(losses['loss'].detach())*n
            totals[3] += n; totals[4] += len(ids)
            step_stats = torch.tensor([float(losses['trajectory_bce'].detach())*n,
                float(losses['group_bce'].detach())*n, float(losses['loss'].detach())*n,
                n, len(ids)], dtype=torch.float64, device=device)
            dist.all_reduce(step_stats)
            global_groups, global_trajectories = int(step_stats[3]), int(step_stats[4])
            if (global_groups != planned['global_questions'] or global_trajectories != global_groups*5):
                raise RuntimeError('Global step diagnostics differ from their real whole-G5 batch')
            global_row = dict(outer=args.outer, step=step, actual_adam_steps_per_rank=actual,
                actual_learning_rate=LR, global_questions=global_groups,
                global_trajectories=global_trajectories,
                trajectory_bce=float(step_stats[0]/global_groups),
                group_bce=float(step_stats[1]/global_groups), total_loss=float(step_stats[2]/global_groups),
                loss_prediction_clock='before_this_step_adam_update', observed_utc=utc())
            global_curve.append(global_row)
            if rank == 0:
                append(out / 'global-loss-updates.jsonl', global_row)
            row = dict(outer=args.outer, rank=rank, step=step, actual_adam_steps=actual,
                actual_learning_rate=LR, local_questions=n, local_trajectories=len(ids),
                global_questions=planned['global_questions'], global_trajectories=planned['global_questions']*5,
                trajectory_bce=float(losses['trajectory_bce'].detach()), group_bce=float(losses['group_bce'].detach()),
                total_loss=float(losses['loss'].detach()), backward_loss=float(backward_loss.detach()),
                loss_semantics='this_step_own_batch_before_update_different_batches_not_monotonic_claim', observed_utc=utc())
            append(out / ('rank-%d-updates.jsonl' % rank), row)
            publish(out / ('progress-rank-%d.json' % rank), dict(row, stage='ONLINE_PROBE_ADAM_WRITE_COMPLETE'))
        train_loop_seconds = time.monotonic()-started
        if totals[3:] != [64, 320]:
            raise RuntimeError('Online probe rank must naturally visit64 own groups/320 trajectories once')
        stats = torch.tensor(totals, dtype=torch.float64, device=device)
        dist.all_reduce(stats)
        if stats[3:].tolist() != [512, 2560]:
            raise RuntimeError('Online natural pass did not cover exactly the selected512G5')
        model.eval()
        predictions = {}
        with torch.no_grad():
            for split, data, rows in (('train', features, groups),
                                      ('test', test, candidate_groups(test.index, 512))):
                local_predictions = []
                own_groups = rows[rank::8]
                for start in range(0, len(own_groups), 6):
                    ids = [tid for group in own_groups[start:start+6] for tid in group['trajectory_ids']]
                    batch = data.get_batch(ids, device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model.module(batch)
                    labels_index = index if split == 'train' else test.index
                    local_predictions.extend((tid, float(z), int(labels_index[tid]['strict_em']))
                                             for tid, z in zip(ids, logits.cpu().tolist()))
                gathered = [None]*8
                dist.all_gather_object(gathered, local_predictions)
                predictions[split] = [row for rows in gathered for row in rows]
        rng = dict(rank=rank, actual_adam_steps=actual_adam_steps(optimizer), python=random.getstate(),
                   numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(device))
        all_rng = [None]*8
        dist.all_gather_object(all_rng, rng)
        if rank == 0:
            train_metrics, train_rows = evaluate_predictions(predictions['train'], index, 512)
            test_metrics, test_rows = evaluate_predictions(predictions['test'], test.index, 512)
            saved_model = {name: value.detach().cpu() for name, value in model.module.state_dict().items()}
            optimizer_state = optimizer.state_dict()
            for state in optimizer_state['state'].values():
                for key, value in list(state.items()):
                    if isinstance(value, torch.Tensor):
                        state[key] = value.detach().cpu()
            checkpoint = dict(schema_version=1, run_id=saved['run_id'], binding=saved['binding'],
                binding_sha256=saved['binding_sha256'], completed_outer=args.outer, model=saved_model,
                optimizer=optimizer_state, rank_rng=all_rng, actual_adam_steps_per_rank=[10*args.outer]*8,
                learning_rate=LR, scored_features_manifest_sha256=features.manifest_sha,
                own_em_sha256=sha(own_path), actor_version=m['actor_version'])
            result = atomic_checkpoint(args.state_dir, checkpoint, args.outer)
            metrics = dict(run_id=saved['run_id'], outer=args.outer, actual_adam_steps_per_rank=[10*args.outer]*8,
                new_adam_steps_per_rank=10, actual_learning_rate=LR, train=train_metrics, test=test_metrics,
                train_online_trajectory_bce=float(stats[0]/512), train_online_group_bce=float(stats[1]/512),
                train_online_total_loss=float(stats[2]/512), train_loop_seconds=train_loop_seconds,
                diagnostic_semantics='same_post_update_head_complete_selected512G5_and_fixed512G5',
                step_loss_semantics='different_natural_batches_before_their_own_update_no_monotonic_claim',
                global_step_loss_curve=summarize_loss_curve(global_curve),
                selected_features_reencoded=False, actor_version=m['actor_version'], checkpoint=result)
            once(out / 'metrics.json', metrics)
            for split, rows in (('train', train_rows), ('test', test_rows)):
                with (out / (split+'-group-predictions.jsonl')).open('x') as handle:
                    for row in rows:
                        handle.write(canonical(row).decode()+'\n')
            once(out / 'complete.json', dict(status='SELECTED512G5_ONLINE_PROBE_10_ADAM_COMPLETE', **metrics,
                                            metrics_sha256=sha(out / 'metrics.json'), observed_utc=utc()))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('initialize', 'score', 'update'))
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--source-checkpoint'); parser.add_argument('--source-sha256')
    parser.add_argument('--run-id'); parser.add_argument('--prefix-inputs'); parser.add_argument('--scored-features')
    parser.add_argument('--own-em'); parser.add_argument('--own-em-sha256'); parser.add_argument('--output')
    parser.add_argument('--outer', type=int); parser.add_argument('--model-dir')
    parser.add_argument('--test-cache'); parser.add_argument('--test-manifest-sha')
    parser.add_argument('--encoder-micro-batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    required = {'initialize': ('source_checkpoint', 'source_sha256', 'run_id', 'test_cache', 'test_manifest_sha'),
                'score': ('prefix_inputs', 'output', 'outer', 'model_dir'),
                'update': ('scored_features', 'own_em', 'output', 'outer', 'model_dir', 'test_cache', 'test_manifest_sha')}
    if any(getattr(args, key) is None for key in required[args.phase]):
        parser.error('Missing phase-specific arguments: '+', '.join(required[args.phase]))
    if args.phase != 'initialize' and not 1 <= args.outer <= 200:
        parser.error('Require outer1..200')
    if args.encoder_micro_batch_size < 1:
        parser.error('Encoder scoring batch must be positive')
    result = {'initialize': initialize, 'score': score, 'update': update}[args.phase](args)
    if result is not None:
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
