"""Four-rank current-actor question+policy5 scoring and periodic G5 probe updates.

The scoring scan has no outcome labels. A qualified, explicit warmup checkpoint
supplies model weights only. Online Adam starts at zero; at outer10,20,...190
it makes50 synchronized writes over all2560 authentic, unselected own G5 groups.
Outer200 deliberately keeps the probe frozen.
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
    batch_to_device, evaluate_predictions,
)
from reproduction.prefix20_online.features import REVISION, canonical, once, read, remote, sha
from reproduction.prefix20_online.launch import publish
from reproduction.prefix20_online.model import expected_policy_positions, validate_prefix_positions
from reproduction.prefix20_online.train_epoch import append, optimizer_learning_rates
from reproduction.prefix20_question_replay.features import FeatureCache, _source_scan
from .contract import (WORLD, QMAX, WIDTH, DIM, PREFIX_TOKENS, G, CANDIDATE_QUESTIONS,
    SELECTED_QUESTIONS, FINAL_OUTER, PROBE_PERIOD, PROBE_UPDATES, PROBE_LR)
from .probe_plan import online_plan, complete_pool_plan, rank_question_slice, ddp_mean_scale
from .candidate_schedule import scoring_candidate_count
from .prefix5_model import (OnlineQuestionProbe, validate_online_layout, warmup_model_weights,
    restore_native_buffers, nonpersistent_buffers, validate_model_state, validate_native_buffers,
    EPOCH3_SHA, WARMUP_RUN)
from reproduction.prefix_only_correctness_ab.train_ddp import actual_adam_steps


SOURCE_SHA = EPOCH3_SHA
SOURCE_RUN = WARMUP_RUN
LR = PROBE_LR
TENSOR_FIELDS = ('features', 'attention_mask', 'position_ids', 'question_mask',
                 'policy_mask', 'question_lengths', 'policy_lengths')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def summarize_loss_curve(curve):
    if len(curve) != PROBE_UPDATES or [row['step'] for row in curve] != list(range(1, PROBE_UPDATES+1)):
        raise ValueError('A periodic2560-G5 epoch has50 actual global loss observations')
    return dict(points=curve, first=curve[0], last=curve[-1],
                delta_end_minus_start={key: curve[-1][key]-curve[0][key]
                                      for key in ('trajectory_bce', 'group_bce', 'total_loss')},
                loss_semantics='each_step_global_true_group_weighted_mean_before_its_own_update',
                batches_differ_between_steps=True, monotonic_decrease_required=False,
                actual_adam_updates_per_rank=PROBE_UPDATES, world_size=WORLD, replica_updates_are_not_summed=True)


def candidate_groups(index, expected_questions=None):
    """Group by authentic question identity, never by consecutive five rows."""
    if expected_questions is None:
        expected_questions = len(index) // G
    if type(expected_questions) is not int or expected_questions < 1:
        raise ValueError('Require a positive declared authentic G5 question count')
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
        raise ValueError('Require every own-G5 trajectory outcome exactly once')
    return {tid: dict(row, strict_em=labels[tid]) for tid, row in index.items()}


def prefix_row(original):
    """A real initial EOS may leave no kept policy tokens; keep its question."""
    if original.get('length', original.get('prefix_length')) != 0:
        row = normalized_row(original, QMAX)
        if not 1 <= row['policy_length'] <= PREFIX_TOKENS:
            raise ValueError('Only the authentic first five policy tokens may enter this probe')
        return row
    initial = original.get('initial_context', {})
    length = initial.get('request_length', original.get('question_length'))
    offset = initial.get('context_offset', original.get('question_context_offset'))
    if (type(length) is not int or not 1 <= length <= QMAX or type(offset) is not int or offset < 0
            or original.get('segments') or original.get('segment_bounds')):
        raise ValueError('Actual zero-policy EOS requires its authentic initial prompt context')
    return dict(original, question_length=length, question_context_offset=offset,
                length=0, policy_length=0, prefix_length=0, segments=[], segment_bounds=[])


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
    from .periodic import is_probe_update
    if completed_outer == 0 or is_probe_update(completed_outer):
        os.link(path, directory / ('head-outer-%04d.pt' % completed_outer))
    return receipt


def initial_online_state(source, binding, *, source_epoch=3, source_model_kind='ema', source_run=SOURCE_RUN):
    """Import qualified learned model/buffers only; online Adam/RNG begin afresh."""
    model, buffers, proof = warmup_model_weights(source, epoch=source_epoch,
        model_kind=source_model_kind, source_run=source_run)
    return dict(schema_version=1, run_id=binding['run_id'], binding=binding, binding_sha256=digest(binding),
                completed_outer=0, model=model, model_nonpersistent_buffers=buffers,
                warmup_model_import=proof, optimizer=None, rank_rng=None,
                actual_adam_steps_per_rank=[0]*WORLD, learning_rate=LR)


def initialize(args):
    directory, source = remote(args.state_dir), remote(args.source_checkpoint)
    source_epoch, model_kind = args.source_epoch, args.source_model_kind
    expected_sha = args.source_sha256
    if (type(source_epoch) is not int or not 1 <= source_epoch <= 200 or model_kind not in ('ema', 'raw')
            or len(expected_sha) != 64 or any(c not in '0123456789abcdef' for c in expected_sha)
            or (source_epoch == 3 and expected_sha != EPOCH3_SHA)):
        raise ValueError('Require the explicit qualified source epoch/SHA/model kind; CP3 has its own exact SHA')
    if (directory / 'initialization-complete.json').exists():
        existing = read(directory / 'initialization-complete.json')
        if (existing['run_id'] != args.run_id or existing['source_sha256'] != expected_sha
                or existing['source_epoch'] != source_epoch or existing['source_model_kind'] != model_kind):
            raise ValueError('Existing online probe belongs to another source/model initialization')
        return dict(status='EXISTING_PROBE_INITIALIZATION_NO_REPLAY', receipt=existing)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError('The requested complete warmup checkpoint must actually exist; historical metadata is insufficient')
    before = source.stat()
    if sha(source) != expected_sha:
        raise ValueError('Requested complete warmup checkpoint bytes differ; another epoch cannot be substituted')
    with source.open('rb') as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
            raise ValueError('Warmup checkpoint changed after its SHA qualification')
        saved = torch.load(handle, map_location='cpu', weights_only=False)
        after = os.fstat(handle.fileno())
    current = source.stat()
    if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or (
            opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
        raise ValueError('Warmup checkpoint changed while loading')
    model, buffers, proof = warmup_model_weights(saved, epoch=source_epoch,
        model_kind=model_kind, source_run=args.source_run)
    source_binding = saved['binding']
    if source_binding['feature_manifest_sha256'] != args.test_manifest_sha:
        raise ValueError('Initial fixed diagnostic source must bind the accepted complete warmup feature manifest')
    binding = dict(role='prefix_selection200_actorhidden_period10_question5_correctness_probe',
        representation_source='current_sampling_actor_native_layer27_postblock',
        independent_encoder_forward=False, probe_update_period=PROBE_PERIOD, updates_per_period=PROBE_UPDATES, run_id=args.run_id,
        encoder_revision=REVISION, source_feature_manifest_sha256=source_binding['feature_manifest_sha256'],
        source_feature_spec_sha256=source_binding['feature_spec_sha256'],
        question_max_tokens=QMAX, head_max_tokens=WIDTH, feature_layer=27, feature_boundary='post_block',
        prefix_tokens=PREFIX_TOKENS, source_probe_sha256=expected_sha, source_epoch=source_epoch,
        source_model_kind=model_kind, source_warmup_run=args.source_run,
        warmup_model_weights_only=True, warmup_optimizer_imported=False,
        online_initial_adam_steps=0, learning_rate=LR,
        group_weight=1.0, loss='mean_trajectory_BCE_plus_mean_G5_mixed_BCE',
        updates_per_outer=0, global_batch_average=256, actual_global_batches=[255, 260],
        world_size=WORLD, accumulation=1, rank_questions_per_period=CANDIDATE_QUESTIONS//WORLD, rank_trajectories_per_period=CANDIDATE_QUESTIONS*G//WORLD,
        diagnostic_manifest_sha256=args.test_manifest_sha, diagnostic_questions=512,
        selected_questions=SELECTED_QUESTIONS, candidate_questions=CANDIDATE_QUESTIONS, exploration=0,
        final_outer_probe_update=False,
        probe_total_actual_adam_per_rank=((FINAL_OUTER-1)//PROBE_PERIOD)*PROBE_UPDATES,
        selection='q_descending_then_original_candidate_ordinal', zero_policy_eos='authentic_question_only_no_fake_token')
    initial = dict(schema_version=1, run_id=binding['run_id'], binding=binding, binding_sha256=digest(binding),
        completed_outer=0, model=model, model_nonpersistent_buffers=buffers, warmup_model_import=proof,
        optimizer=None, rank_rng=None, actual_adam_steps_per_rank=[0]*WORLD, learning_rate=LR)
    directory.mkdir(parents=True, exist_ok=True)
    once(directory / 'initialization-intent.json', dict(run_id=args.run_id, source=str(source),
        source_sha256=expected_sha, source_epoch=source_epoch, source_model_kind=model_kind))
    receipt = atomic_checkpoint(directory, initial, 0)
    result = dict(status='QUALIFIED_QUESTION5_WARMUP_MODEL_ONLY_ONLINE_ADAM0_INITIALIZED', run_id=args.run_id,
        source_sha256=expected_sha, source_epoch=source_epoch, source_model_kind=model_kind,
        source_model_tensors=14, native_buffers_preserved=True, warmup_optimizer_imported=False,
        warmup_rng_imported=False, warmup_updates_reexecuted=0, source_import=proof, receipt=receipt)
    once(directory / 'initialization-complete.json', result)
    return result


def setup_ddp():
    rank, local, world = (int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != WORLD or torch.cuda.device_count() != WORLD:
        raise ValueError('Probe phases require the actual four CUDA ranks')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    return rank, local, torch.device('cuda', local)


def load_online(args, device):
    directory = remote(args.state_dir)
    receipt = read(directory / 'latest.json')
    from .periodic import probe_boundary
    expected = probe_boundary(args.outer-1)
    if (receipt['completed_outer'] != expected['last_update_outer'] or
            receipt['actual_adam_steps_per_rank'] != [expected['actual_adam_per_rank']]*WORLD):
        raise ValueError('Probe must be the last completed10-outer state (outer200 retains190)')
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
            or saved['completed_outer'] != expected['last_update_outer'] or saved['actual_adam_steps_per_rank'] != receipt['actual_adam_steps_per_rank']
            or saved['binding']['learning_rate'] != LR):
        raise ValueError('Online probe binding or actual Adam boundary differs')
    model = OnlineQuestionProbe(remote(args.model_dir), QMAX).to(device=device, dtype=torch.float32)
    validate_model_state(saved['model'])
    validate_native_buffers(saved['model_nonpersistent_buffers'])
    model.load_state_dict(saved['model'], strict=True)
    restore_native_buffers(model, saved['model_nonpersistent_buffers'])
    if (saved['binding'].get('world_size') != WORLD or saved['binding'].get('prefix_tokens') != PREFIX_TOKENS
            or saved['binding'].get('question_max_tokens') != QMAX or saved['binding'].get('head_max_tokens') != WIDTH):
        raise ValueError('Online probe source architecture or four-rank contract differs')
    return model, saved, receipt


def score(args):
    from .native_features import NativeFeatures
    rank, local, device = setup_ddp()
    features = None
    try:
        model, saved, receipt = load_online(args, device)
        features = NativeFeatures(args.prefix_inputs)
        manifest = features.manifest
        actor_version = manifest['actor_version']
        if manifest['outer'] != args.outer or manifest['run_id'] != saved['run_id']:
            raise ValueError('Native features belong to another actual actor/outer/run')
        questions = scoring_candidate_count(manifest)
        groups = candidate_groups(features.index, questions)
        own_groups = groups[rank::WORLD]
        out = remote(args.output)
        if rank == 0: out.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        local_logits, fallbacks = [], []
        model.eval(); started=time.monotonic()
        group_batch=max(1,args.encoder_micro_batch_size//5)
        with torch.no_grad():
            for offset in range(0,len(own_groups),group_batch):
                ids=[tid for g in own_groups[offset:offset+group_batch] for tid in g['trajectory_ids']]
                batch=features.get_batch(ids,device)
                logits=inference_logits(model,batch,args.encoder_micro_batch_size,fallbacks)
                local_logits.extend(zip(ids,logits.cpu().tolist()))
                publish(out/('score-progress-rank-%d.json'%rank),dict(outer=args.outer,
                    scored_trajectories=len(local_logits),stage='CURRENT_ACTOR_NATIVE_HIDDEN_HEAD_SCORING'))
        gathered=[None]*WORLD;dist.all_gather_object(gathered,local_logits)
        merged=[pair for rows in gathered for pair in rows]
        if len(merged)!=questions*G or len({tid for tid,_ in merged})!=questions*G:
            raise ValueError('Score every scheduled authentic native G5 exactly once')
        selected,all_scores=stable_top512(groups,dict(merged))
        if rank==0:
            common=dict(schema_version=2,run_id=saved['run_id'],outer=args.outer,actor_version=actor_version,
                prefix_scan_sha256=features.manifest_sha,probe_before_checkpoint_sha256=receipt['sha256'],
                candidate_questions=questions,trajectories_per_question=G,exploration=0,
                formula='1-prod(p_j)-prod(1-p_j)',outcome_labels_used=False,
                representation_source='same_sampling_actor_native_layer27_postblock',
                **{key: manifest[key] for key in ('candidate_schedule', 'scheduled_candidate_questions') if key in manifest})
            from .periodic import is_probe_update
            selection=dict(common,selected=selected,selected_questions=SELECTED_QUESTIONS,
                tie_break='original_candidate_ordinal_ascending',continuation_scope='top512',
                provisional_for_actor=is_probe_update(args.outer),
                selection_semantics='predicted_mixed_probability_top512_before_any_outcome')
            once(out/'selection.json',selection)
            from .periodic import is_probe_update
            if is_probe_update(args.outer):
                once(out/'pool-selection.json',dict(common,selected=sorted(all_scores,key=lambda r:r['candidate_ordinal']),
                     selected_questions=questions,continuation_scope='all_candidates'))
            with (out/'candidate-scores.jsonl').open('x') as f:
                for row in all_scores:f.write(canonical(row).decode()+'\n')
            once(out/'index.json',list(features.index.values()))
            result=dict(role='current_actor_native_hidden_features',run_id=saved['run_id'],outer=args.outer,
                count=questions*G,selected_count=SELECTED_QUESTIONS*G,question_max_tokens=QMAX,head_max_tokens=WIDTH,
                actor_version=actor_version,prefix_scan_sha256=features.manifest_sha,
                probe_before_checkpoint_sha256=receipt['sha256'],selection_sha256=sha(out/'selection.json'),
                native_hidden_scan=str(features.path),native_hidden_shards=manifest['actor_hidden_shards'],
                index_sha256=sha(out/'index.json'),candidate_scores_sha256=sha(out/'candidate-scores.jsonl'),
                no_outcome_labels=True,independent_encoder_forward=False,
                representation_source='same_sampling_actor_native_layer27_postblock',
                candidate_questions=questions,
                **{key: manifest[key] for key in ('candidate_schedule', 'scheduled_candidate_questions') if key in manifest},
                elapsed_seconds=time.monotonic()-started)
            once(out/'manifest.json',result)
            once(out/'complete.json',dict(status='PREFIX%dG5_STABLE_TOP512_SCORE_COMPLETE' % questions,outer=args.outer,
                run_id=saved['run_id'],manifest_sha256=sha(out/'manifest.json'),
                selection_sha256=sha(out/'selection.json'),selected_questions=512,selected_trajectories=2560,
                candidate_questions=questions,
                actor_version=actor_version,observed_utc=utc()))
        dist.barrier()
    finally:
        if features is not None:features.close()
        if dist.is_initialized():dist.destroy_process_group()


def update(args):
    import numpy as np
    rank, local, device = setup_ddp()
    try:
        model, saved, receipt = load_online(args, device)
        from .native_features import NativeFeatures
        from .periodic import is_probe_update, probe_boundary
        if not is_probe_update(args.outer):
            raise ValueError('Probe optimization only at10,20,...190; outer200 is frozen')
        m=read(remote(args.scored_features)/'manifest.json')
        features=NativeFeatures(m['native_hidden_scan'])
        if (m['run_id']!=saved['run_id'] or m['outer']!=args.outer
                or m['probe_before_checkpoint_sha256']!=receipt['sha256']
                or features.manifest['actor_version']!=m['actor_version']):
            raise ValueError('Periodic pool hidden/actor/probe provenance differs')
        own_path=remote(args.own_em)
        if args.own_em_sha256 and sha(own_path)!=args.own_em_sha256:
            raise ValueError('Complete unselected pool outcomes differ from receipt')
        records=[json.loads(line) for line in own_path.open()]
        index=join_own_em(features.index,records,m['actor_version'],m['prefix_scan_sha256'],m['selection_sha256'])
        groups=candidate_groups(index,CANDIDATE_QUESTIONS)
        test=NativeFeatures(args.test_cache)
        if test.manifest_sha!=args.test_manifest_sha or test.manifest['actor_version']!=m['actor_version']:
            raise ValueError('Diagnostic hidden must come from this exact sampling actor')
        test_outcomes=Path(args.test_cache).parent.parent/'full/own-em.jsonl'
        test_index=join_own_em(test.index,[json.loads(line) for line in test_outcomes.open()],
            m['actor_version'],test.manifest_sha,'diagnostic')
        test.index=test_index
        if ({str(row['qid']) for row in index.values()} & {str(row['qid']) for row in test.index.values()}
                or {row['question_hash'] for row in index.values()} & {row['question_hash'] for row in test.index.values()}):
            raise ValueError('Periodic2560 training pool overlaps fixed512 diagnostic')
        def evaluate(head):
            head.eval();result={}
            with torch.no_grad():
                for split,data,rows in (('train',features,groups),('test',test,candidate_groups(test.index,512))):
                    local_predictions=[]
                    own_groups=rows[rank::WORLD]
                    for start in range(0,len(own_groups),6):
                        ids=[tid for group in own_groups[start:start+6] for tid in group['trajectory_ids']]
                        batch=data.get_batch(ids,device)
                        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                            logits=head.module(batch) if isinstance(head,DistributedDataParallel) else head(batch)
                        labels_index=index if split=='train' else test.index
                        local_predictions.extend((tid,float(z),int(labels_index[tid]['strict_em']))
                            for tid,z in zip(ids,logits.cpu().tolist()))
                    gathered=[None]*WORLD;dist.all_gather_object(gathered,local_predictions)
                    result[split]=[row for rows in gathered for row in rows]
            return result
        before_predictions=evaluate(model)
        before_metrics={}
        if rank==0:
            before_metrics={split:evaluate_predictions(before_predictions[split],
                index if split=='train' else test.index,CANDIDATE_QUESTIONS if split=='train' else SELECTED_QUESTIONS)[0]
                for split in ('train','test')}
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
        expected_before = probe_boundary(args.outer-1)['actual_adam_per_rank']
        if expected_before:
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
        plan = complete_pool_plan(args.seed, args.outer)
        binding_values = [None]*WORLD
        dist.all_gather_object(binding_values, (features.manifest_sha, sha(own_path), digest(plan), receipt['sha256']))
        if len(set(binding_values)) != 1:
            raise ValueError('Four ranks differ in features, outcomes, plan, or prior probe')
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
        if totals[3:] != [CANDIDATE_QUESTIONS//WORLD, CANDIDATE_QUESTIONS*G//WORLD]:
            raise RuntimeError('Periodic rank must visit640 own groups/3200 trajectories once')
        stats = torch.tensor(totals, dtype=torch.float64, device=device)
        dist.all_reduce(stats)
        if stats[3:].tolist() != [CANDIDATE_QUESTIONS, CANDIDATE_QUESTIONS*G]:
            raise RuntimeError('Periodic natural pass must cover exactly the unselected2560G5')
        predictions = evaluate(model)
        rng = dict(rank=rank, actual_adam_steps=actual_adam_steps(optimizer), python=random.getstate(),
                   numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(device))
        all_rng = [None]*WORLD
        dist.all_gather_object(all_rng, rng)
        if rank == 0:
            train_metrics, train_rows = evaluate_predictions(predictions['train'], index, CANDIDATE_QUESTIONS)
            test_metrics, test_rows = evaluate_predictions(predictions['test'], test.index, 512)
            saved_model = {name: value.detach().cpu() for name, value in model.module.state_dict().items()}
            optimizer_state = optimizer.state_dict()
            for state in optimizer_state['state'].values():
                for key, value in list(state.items()):
                    if isinstance(value, torch.Tensor):
                        state[key] = value.detach().cpu()
            checkpoint = dict(schema_version=1, run_id=saved['run_id'], binding=saved['binding'],
                binding_sha256=saved['binding_sha256'], completed_outer=args.outer, model=saved_model,
                model_nonpersistent_buffers=nonpersistent_buffers(model.module),
                warmup_model_import=saved.get('warmup_model_import'), optimizer=optimizer_state, rank_rng=all_rng, actual_adam_steps_per_rank=[probe_boundary(args.outer)['actual_adam_per_rank']]*WORLD,
                learning_rate=LR, scored_features_manifest_sha256=features.manifest_sha,
                own_em_sha256=sha(own_path), actor_version=m['actor_version'])
            result = atomic_checkpoint(args.state_dir, checkpoint, args.outer)
            metrics = dict(run_id=saved['run_id'], outer=args.outer, actual_adam_steps_per_rank=[probe_boundary(args.outer)['actual_adam_per_rank']]*WORLD,
                new_adam_steps_per_rank=PROBE_UPDATES, actual_learning_rate=LR, train=train_metrics, test=test_metrics,
                before_update=before_metrics, unselected_training_questions=CANDIDATE_QUESTIONS, actor_hidden_native=True,
                train_online_trajectory_bce=float(stats[0]/CANDIDATE_QUESTIONS), train_online_group_bce=float(stats[1]/CANDIDATE_QUESTIONS),
                train_online_total_loss=float(stats[2]/CANDIDATE_QUESTIONS), train_loop_seconds=train_loop_seconds,
                diagnostic_semantics='same_head_pre_and_post_update_full2560G5_and_current_actor_fixed512G5',
                step_loss_semantics='different_natural_batches_before_their_own_update_no_monotonic_claim',
                global_step_loss_curve=summarize_loss_curve(global_curve),
                selected_features_reencoded=False, actor_version=m['actor_version'], checkpoint=result,
                scored_features_manifest_sha256=features.manifest_sha,
                own_em_sha256=sha(own_path), selection_sha256=m['selection_sha256'])
            once(out / 'metrics.json', metrics)
            for split, rows in (('train', train_rows), ('test', test_rows)):
                with (out / (split+'-group-predictions.jsonl')).open('x') as handle:
                    for row in rows:
                        handle.write(canonical(row).decode()+'\n')
            once(out / 'complete.json', dict(status='UNSELECTED2560G5_PERIODIC_PROBE_50_ADAM_COMPLETE', **metrics,
                                            metrics_sha256=sha(out / 'metrics.json'), observed_utc=utc()))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('initialize', 'score', 'update'))
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--source-checkpoint'); parser.add_argument('--source-sha256', default=EPOCH3_SHA)
    parser.add_argument('--source-epoch', type=int, default=3)
    parser.add_argument('--source-model-kind', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--source-run', default=SOURCE_RUN)
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
