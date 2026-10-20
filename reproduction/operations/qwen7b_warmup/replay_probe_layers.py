"""Bounded remote-GPU recapture of sealed actual7B Prefix20 token segments.

This generates NO tokens and starts NO Slurm job.  It replays each saved causal
segment once with frozen HF official7B and retains postblocks20/23/26 from
hidden_states[21/24/27].  Question slots come only from turn0; later tool
contexts contribute their own retained policy slots at their actual positions.

HF prefill and vLLM native sampling can differ numerically.  Qualification
records retained-slot L26 comparison metrics, never claims bit-exact equality,
and never automatically accepts this backend for training.  This CLI is
deliberately limited to at most128 explicitly declared fixture trajectories.
All model, native reference, replay features and outputs must stay remote.

Selection JSON schema:
  model_revision, prefix_files:[{path,bytes,sha256}],
  trajectories:[{trajectory_id,qid}],
  native_layer26_manifests:[{path,bytes,sha256}],
  model_files:{filename:{bytes,sha256}}.
Model files must cover official-inventory.json, config.json, the safetensors
index, and every shard referenced by that index.  The selection JSON itself is
bound by the mandatory --selection-sha256 flag.
"""
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import zlib


MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
LAYERS = (20, 23, 26)
QMAX, PREFIX, DIM = 273, 20, 3584
WIDTH = QMAX+PREFIX
CAPTURE_KIND = 'post_block_residual'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def fingerprints(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def remote(path):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or
            not str(path).startswith(('/data/', '/dataset/', '/backup/', '/mnt/'))):
        raise ValueError('Explicit nonsymlink remote artifact path required: '+str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(value)+b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer '+name)
    return value


def tokens(value, name):
    if not isinstance(value, list) or any(type(v) is not int or v < 0 for v in value):
        raise ValueError('Actual nonnegative integer token IDs required: '+name)
    return list(value)


def normalize_prefix(row, expected_tid=None, expected_qid=None, require_digest=False,
                     max_context_tokens=4096):
    """Pure validation; never reconstruct requests by decoding/re-tokenizing."""
    tid, qid = row['trajectory_id'], str(row['qid'])
    if not isinstance(tid, str) or not tid:
        raise ValueError('A real trajectory identity is required')
    if type(row['qid']) not in (str, int) or not qid:
        raise ValueError('A real question identity is required')
    if expected_tid is not None and tid != expected_tid:
        raise ValueError('Prefix trajectory identity differs from selection')
    if expected_qid is not None and qid != str(expected_qid):
        raise ValueError('Prefix question identity differs from selection')
    if row['actor_version'] != 'official7b-'+REVISION:
        raise ValueError('Prefix actor is not the frozen official7B snapshot')
    if row.get('model_revision', REVISION) != REVISION:
        raise ValueError('Prefix model revision differs')
    if row.get('feature_actor_version', row['actor_version']) != row['actor_version']:
        raise ValueError('Original native feature actor differs from the sealed prefix actor')
    if 'feature_layer' in row and row['feature_layer'] != 26:
        raise ValueError('Original qualification reference must be native layer26')
    if row.get('future_information_included') is not False:
        raise ValueError('Source prefix must explicitly exclude future information')
    if require_digest or 'prefix_sha256' in row:
        # Native sampling adds feature_* locators AFTER sealing prefix_sha256.
        body = {k:v for k,v in row.items() if k != 'prefix_sha256' and not k.startswith('feature_')}
        if hashlib.sha256(canonical(body)).hexdigest() != row.get('prefix_sha256'):
            raise ValueError('Sealed prefix content digest differs')
    prompt = tokens(row['initial_prompt_token_ids'], 'initial prompt')
    policy = tokens(row['policy_token_ids'], 'own retained policy')
    positions = tokens(row['policy_position_ids'], 'own native policy positions')
    if not 1 <= len(prompt) <= QMAX or not 0 <= len(policy) <= PREFIX:
        raise ValueError('Retained prompt/Prefix20 width differs')
    if integer(row['retained_policy_length'], 'retained_policy_length') != len(policy):
        raise ValueError('Retained policy count differs')
    if len(positions) != len(policy):
        raise ValueError('Every retained policy token needs its own native position')
    if len(policy) < PREFIX and row.get('terminal') is not True:
        raise ValueError('A short prefix requires a real terminal boundary')
    raw_segments = row['prefix_segments']
    if not isinstance(raw_segments, list):
        raise ValueError('Saved actual prefix segments required')
    segments, offset, previous_turn, previous_context, expected_positions = [], 0, -1, None, []
    for original in raw_segments:
        segment = dict(original)
        turn = integer(segment['turn_index'], 'turn_index')
        start = integer(segment['policy_start'], 'policy_start')
        length = integer(segment['policy_length'], 'policy_length', 1)
        request = integer(segment['request_length'], 'request_length', 1)
        position = integer(segment['position_start'], 'position_start', 1)
        context = tokens(segment['context_tokens'], 'actual causal context')
        if turn <= previous_turn or start != offset or start+length > len(policy):
            raise ValueError('Policy segment coverage overlaps, skips or exceeds own retained prefix')
        if request != position or len(context) != request+length or len(context) > max_context_tokens:
            raise ValueError('Actual causal request length/position/context differs; future tokens are forbidden')
        if context[request:] != policy[start:start+length]:
            raise ValueError('Segment contains wrong trajectory policy tokens or future generated tokens')
        if not segments:
            if turn != 0 or context[:request] != prompt:
                raise ValueError('Original turn0 question request is missing or changed')
        elif context[:len(prompt)] != prompt or context[:len(previous_context)] != previous_context:
            raise ValueError('Later tool context does not preserve its actual earlier causal input')
        segment.update(context_tokens=context, context_length=len(context), capture_question=not segments)
        segments.append(segment)
        expected_positions.extend(range(position, position+length))
        offset, previous_turn, previous_context = start+length, turn, context
    if offset != len(policy) or expected_positions != positions:
        raise ValueError('Saved segments and native retained token/position coverage differ')
    if not policy:
        if raw_segments:
            raise ValueError('Empty policy cannot contain generated-token segments')
        segments = [dict(turn_index=0, policy_start=0, policy_length=0,
                         request_length=len(prompt), position_start=len(prompt),
                         context_tokens=prompt, context_length=len(prompt), capture_question=True)]
    return dict(trajectory_id=tid, qid=qid, initial_prompt_token_ids=prompt,
                policy_token_ids=policy, policy_position_ids=positions,
                question_length=len(prompt), policy_length=len(policy), segments=segments,
                source_prefix_sha256=row.get('prefix_sha256'), actor_version=row['actor_version'],
                source_native_manifest=row.get('feature_file'))


def replay_features(normalized, capture, hidden_size=DIM):
    """CPU-testable alignment around one supplied causal multi-layer capture."""
    import torch
    result = {layer:torch.zeros(WIDTH, hidden_size, dtype=torch.bfloat16) for layer in LAYERS}
    filled = torch.zeros(WIDTH, dtype=torch.bool)
    qlen, plen = normalized['question_length'], normalized['policy_length']
    positions = torch.zeros(WIDTH, dtype=torch.long)
    positions[:qlen] = torch.arange(qlen)
    positions[QMAX:QMAX+plen] = torch.tensor(normalized['policy_position_ids'], dtype=torch.long)
    for segment in normalized['segments']:
        context = list(segment['context_tokens'])
        # No complete future response, sampled lookahead, or tokens from another
        # segment are supplied.  Native RoPE positions are exactly0..N-1.
        hidden = capture(context, list(range(len(context))))
        if set(hidden) != set(LAYERS):
            raise ValueError('Every causal segment must capture all three requested layers in one pass')
        request, length, start = segment['request_length'], segment['policy_length'], segment['policy_start']
        for layer in LAYERS:
            own = hidden[layer]
            if own.shape != (len(context), hidden_size) or own.requires_grad:
                raise ValueError('Frozen causal capture returned invalid shape/gradient state')
            if not bool(torch.isfinite(own).all()):
                raise ValueError('Frozen capture contains nonfinite hidden values')
            own = own.detach().to(device='cpu', dtype=torch.bfloat16, copy=True)
            if segment['capture_question']:
                if bool(filled[:qlen].any()):
                    raise ValueError('Question slots must only be captured from initial turn0 context')
                result[layer][:qlen].copy_(own[:qlen])
            result[layer][QMAX+start:QMAX+start+length].copy_(own[request:request+length])
        if segment['capture_question']:
            filled[:qlen] = True
        if bool(filled[QMAX+start:QMAX+start+length].any()):
            raise ValueError('A retained policy feature slot was captured twice')
        filled[QMAX+start:QMAX+start+length] = True
    expected = torch.zeros_like(filled)
    expected[:qlen] = True
    expected[QMAX:QMAX+plen] = True
    if not torch.equal(filled, expected):
        raise ValueError('Recaptured slots do not exactly cover initial question and own Prefix20')
    return {layer:dict(features=value, attention_mask=expected.clone(), position_ids=positions.clone(),
                       question_length=qlen, policy_length=plen, capture_layer=layer,
                       capture_kind=CAPTURE_KIND) for layer, value in result.items()}


class ReplayLayerWriter:
    """Stream arbitrary declared fixture counts; no old five-completions rule."""
    def __init__(self, directory, layer, selected, dimension=DIM):
        if layer not in LAYERS:
            raise ValueError('Unexpected replay capture layer')
        self.root, self.layer, self.dimension = Path(directory), layer, dimension
        self.root.mkdir(parents=True, exist_ok=False)
        self.allowed = {r['trajectory_id']:str(r['qid']) for r in selected}
        if len(self.allowed) != len(selected) or not selected:
            raise ValueError('A unique nonempty declared trajectory fixture is required')
        self.remaining = Counter(self.allowed.values())
        self.done, self.questions, self.prototype = set(), {}, None
        self.streams = {name:(self.root/name).open('xb') for name in
                        ('question.bin', 'delta.bin', 'policy.bin', 'index.jsonl')}
        self.hashes = {name:hashlib.sha256() for name in self.streams}

    def put(self, name, payload):
        offset = self.streams[name].tell()
        self.streams[name].write(payload)
        self.hashes[name].update(payload)
        return offset

    def write(self, row, data):
        import numpy as np
        import torch
        tid, qid = row['trajectory_id'], str(row['qid'])
        if tid not in self.allowed or self.allowed[tid] != qid or tid in self.done:
            raise ValueError('Wrong, extra or duplicate replay trajectory')
        if data['capture_layer'] != self.layer or data['capture_kind'] != CAPTURE_KIND:
            raise ValueError('Layer writer received another representation')
        values = data['features']
        qlen, plen = row['question_length'], row['policy_length']
        if (values.device.type != 'cpu' or values.requires_grad or values.dtype != torch.bfloat16 or
                tuple(values.shape) != (WIDTH, self.dimension)):
            raise ValueError('Ordinary detached CPU BF16 retained features required')
        expected_mask = torch.zeros(WIDTH, dtype=torch.bool)
        expected_mask[:qlen] = True
        expected_mask[QMAX:QMAX+plen] = True
        expected_positions = torch.zeros(WIDTH, dtype=torch.long)
        expected_positions[:qlen] = torch.arange(qlen)
        expected_positions[QMAX:QMAX+plen] = torch.tensor(row['policy_position_ids'], dtype=torch.long)
        if (data['question_length'] != qlen or data['policy_length'] != plen or
                not torch.equal(data['attention_mask'], expected_mask) or
                not torch.equal(data['position_ids'], expected_positions)):
            raise ValueError('Packed replay features must retain exactly their own token/position slots')
        if not bool(torch.isfinite(values[expected_mask]).all()):
            raise ValueError('Nonfinite real retained feature')
        raw = values.contiguous().view(torch.uint16).numpy()
        question = raw[:qlen].copy()
        prompt_sha = hashlib.sha256(canonical(row['initial_prompt_token_ids'])).hexdigest()
        if self.prototype is None:
            self.prototype = np.zeros((QMAX, self.dimension), dtype=np.uint16)
            self.prototype[:qlen] = question
            with (self.root/'prototype.bin').open('xb') as stream:
                stream.write(self.prototype.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
        if qid not in self.questions:
            encoded = zlib.compress(np.bitwise_xor(question, self.prototype[:qlen]).tobytes(), 1)
            offset = self.put('question.bin', encoded)
            self.questions[qid] = dict(array=question, offset=offset, bytes=len(encoded),
                                        qlen=qlen, prompt_sha256=prompt_sha)
        base = self.questions[qid]
        if base['qlen'] != qlen or base['prompt_sha256'] != prompt_sha:
            raise ValueError('Question identity changed within a replay fixture')
        delta = np.bitwise_xor(question, base['array'])
        encoded_delta = zlib.compress(delta.tobytes(), 1) if bool(np.any(delta)) else b''
        delta_offset = self.put('delta.bin', encoded_delta)
        policy = raw[QMAX:QMAX+plen].copy()
        policy_offset = self.put('policy.bin', policy.tobytes())
        restored = base['array'] if not encoded_delta else np.bitwise_xor(base['array'],
            np.frombuffer(zlib.decompress(encoded_delta), dtype=np.uint16).reshape(question.shape))
        if not np.array_equal(restored, question):
            raise ValueError('Replay question XOR packing is not lossless')
        item = dict(trajectory_id=tid, qid=qid, question_length=qlen, policy_length=plen,
            question_offset=base['offset'], question_bytes=base['bytes'],
            delta_offset=delta_offset, delta_bytes=len(encoded_delta), policy_offset=policy_offset,
            dimension=self.dimension, policy_positions=row['policy_position_ids'],
            prompt_sha256=prompt_sha, policy_token_sha256=hashlib.sha256(canonical(row['policy_token_ids'])).hexdigest(),
            source_prefix_sha256=row['source_prefix_sha256'], capture_layer=self.layer,
            capture_kind=CAPTURE_KIND, retained_feature_sha256=hashlib.sha256(question.tobytes()+policy.tobytes()).hexdigest())
        self.put('index.jsonl', canonical(item)+b'\n')
        self.done.add(tid)
        self.remaining[qid] -= 1
        if self.remaining[qid] == 0:
            del self.questions[qid]

    def finish(self):
        if self.done != set(self.allowed) or self.questions or any(self.remaining.values()):
            raise ValueError('Replay writer is missing declared trajectories')
        files = {}
        for name, stream in self.streams.items():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            files[name] = dict(path=str(self.root/name), bytes=(self.root/name).stat().st_size,
                               sha256=self.hashes[name].hexdigest())
        files['prototype.bin'] = fingerprints(self.root/'prototype.bin')
        result = dict(status='LOSSLESS_HF_REPLAY_FEATURES_COMPLETE', count=len(self.done),
            capture_layer=self.layer, feature_layer=self.layer, capture_kind=CAPTURE_KIND,
            model=MODEL, model_revision=REVISION, dimension=self.dimension,
            question_max_tokens=QMAX, prefix_tokens=PREFIX, files=files,
            layout='lossless_question_base_xor_policy_v1', source_runtime='huggingface_frozen_causal_prefill',
            bit_exact_native_vllm_claimed=False, actor_optimizer_updates=0)
        atomic(self.root/'complete.json', result)
        return fingerprints(self.root/'complete.json')


class Comparison:
    """Small aggregate metrics; never return raw native/replay feature tensors."""
    def __init__(self):
        self.elements = self.tokens = self.valid_cosines = 0
        self.absolute = self.squared_error = self.reference_squared = self.replay_squared = self.dot = 0.
        self.maximum_absolute = 0.
        self.cosine_sum, self.minimum_cosine = 0., 1.

    def add(self, replay, reference):
        import torch
        if replay.shape != reference.shape or replay.ndim != 2:
            raise ValueError('Numerical qualification requires aligned retained token vectors')
        a, b = replay.detach().cpu().double(), reference.detach().cpu().double()
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError('Nonfinite numerical qualification input')
        if not a.numel():
            return
        diff = a-b
        self.elements += a.numel()
        self.tokens += len(a)
        self.absolute += float(diff.abs().sum())
        self.maximum_absolute = max(self.maximum_absolute, float(diff.abs().max()))
        self.squared_error += float(diff.square().sum())
        self.reference_squared += float(b.square().sum())
        self.replay_squared += float(a.square().sum())
        self.dot += float((a*b).sum())
        denominator = a.norm(dim=1)*b.norm(dim=1)
        valid = denominator > 0
        if bool(valid.any()):
            cosine = ((a*b).sum(1)[valid]/denominator[valid]).clamp(-1, 1)
            self.valid_cosines += len(cosine)
            self.cosine_sum += float(cosine.sum())
            self.minimum_cosine = min(self.minimum_cosine, float(cosine.min()))

    def result(self):
        denominator = math.sqrt(self.reference_squared*self.replay_squared)
        return dict(tokens=self.tokens, elements=self.elements,
            mean_absolute_error=self.absolute/self.elements if self.elements else None,
            max_absolute_error=self.maximum_absolute if self.elements else None,
            relative_l2=math.sqrt(self.squared_error/self.reference_squared) if self.reference_squared else
                        (0. if self.elements and self.squared_error == 0 else None),
            global_cosine=max(-1., min(1., self.dot/denominator)) if denominator else None,
            mean_token_cosine=self.cosine_sum/self.valid_cosines if self.valid_cosines else None,
            min_token_cosine=self.minimum_cosine if self.valid_cosines else None,
            valid_nonzero_cosine_tokens=self.valid_cosines)


def checked_file(descriptor):
    path = remote(descriptor['path'])
    actual = fingerprints(path)
    if actual['sha256'] != descriptor['sha256'] or actual['bytes'] != descriptor['bytes']:
        raise ValueError('SHA/size mismatch for declared source: '+str(path))
    return actual


def load_selected(selection, maximum, max_context_tokens):
    declared = selection['trajectories']
    if not declared or len(declared) > maximum or maximum > 128:
        raise ValueError('Qualification requires a declared small fixture, not a full-bank recapture')
    wanted = {row['trajectory_id']:str(row['qid']) for row in declared}
    if len(wanted) != len(declared):
        raise ValueError('Duplicate declared trajectory')
    found, source_proofs = {}, []
    for descriptor in selection['prefix_files']:
        source_proofs.append(checked_file(descriptor))
        path = Path(descriptor['path'])
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                tid = row['trajectory_id']
                if tid not in wanted:
                    continue
                if tid in found:
                    raise ValueError('Requested trajectory occurs more than once in sealed prefix sources')
                found[tid] = normalize_prefix(row, tid, wanted[tid], True, max_context_tokens)
    if set(found) != set(wanted):
        raise ValueError('A requested trajectory is missing from the sealed prefix sources')
    selected = [found[row['trajectory_id']] for row in declared]
    prompts = {}
    for row in selected:
        previous = prompts.setdefault(row['qid'], row['initial_prompt_token_ids'])
        if previous != row['initial_prompt_token_ids']:
            raise ValueError('Question prompt changed across declared sibling trajectories')
    return selected, source_proofs


def verify_model(model_dir, selection):
    root = remote(model_dir)
    declared = selection['model_files']
    needed = {'official-inventory.json', 'config.json', 'model.safetensors.index.json'}
    if not needed.issubset(declared):
        raise ValueError('Official model inventory/config/index must be SHA-bound')
    proofs = {}
    for name in needed:
        proofs[name] = checked_file(dict(declared[name], path=str(root/name)))
    inventory = read(root/'official-inventory.json')
    if inventory['sha'] != REVISION or inventory.get('id', MODEL) != MODEL:
        raise ValueError('Wrong official model snapshot inventory')
    config = read(root/'config.json')
    if (config['num_hidden_layers'], config['hidden_size'], config['num_attention_heads'],
            config['num_key_value_heads'], config['intermediate_size']) != (28, 3584, 28, 4, 18944):
        raise ValueError('Replay requires official28-layer/3584-wide Qwen7B')
    shards = set(read(root/'model.safetensors.index.json')['weight_map'].values())
    if not shards.issubset(declared) or any(Path(name).name != name for name in shards):
        raise ValueError('Every actual official model shard must have a source SHA')
    official = {item['rfilename']:item for item in inventory['siblings']}
    for name in ('config.json', 'model.safetensors.index.json'):
        source = official[name]
        if source.get('lfs'):
            if proofs[name]['sha256'] != source['lfs']['sha256']:
                raise ValueError('Official model metadata LFS SHA differs')
        else:
            payload = (root/name).read_bytes()
            actual_blob = hashlib.sha1(('blob %d\0' % len(payload)).encode()+payload).hexdigest()
            if actual_blob != source['blobId']:
                raise ValueError('Official model metadata Git blob differs')
    for name in sorted(shards):
        proofs[name] = checked_file(dict(declared[name], path=str(root/name)))
        source = official[name]
        if 'lfs' not in source or proofs[name]['sha256'] != source['lfs']['sha256']:
            raise ValueError('Model shard differs from pinned official LFS SHA')
    return proofs, config


def verify_native_sources(selection):
    proofs, manifests = [], []
    for descriptor in selection['native_layer26_manifests']:
        proofs.append(checked_file(descriptor))
        path = Path(descriptor['path'])
        item = read(path)
        if (item['status'] != 'LOSSLESS_NATIVE_FEATURES_COMPLETE' or item['feature_layer'] != 26 or
                item['dimension'] != DIM or item['question_max_tokens'] != QMAX or item['prefix_tokens'] != PREFIX):
            raise ValueError('Reference must be authentic retained native-vLLM layer26 features')
        if not {'question.bin', 'delta.bin', 'policy.bin', 'index.jsonl', 'prototype.bin'}.issubset(item['files']):
            raise ValueError('Native packed reference is missing required source files')
        for name, source in item['files'].items():
            if Path(source['path']).resolve() != path.parent/name or Path(name).name != name:
                raise ValueError('Native reference file locator does not match the actual reader source')
            proofs.append(checked_file(source))
        manifests.append(str(path))
    if not manifests:
        raise ValueError('Small-fixture nativeL26 qualification reference is mandatory')
    return manifests, proofs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-trajectories', type=int, default=32)
    parser.add_argument('--max-context-tokens', type=int, default=4096)
    parser.add_argument('--attention-implementation', choices=('flash_attention_2', 'sdpa', 'eager'), default='flash_attention_2')
    args = parser.parse_args()
    import torch
    if platform.system() != 'Linux' or not torch.cuda.is_available():
        raise RuntimeError('Actual remote Linux CUDA execution is mandatory; no local model loading')
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Use the already authorized Slurm GPU allocation, not an unowned device')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Frozen replay must use an allocated CUDA device')
    torch.cuda.set_device(device)
    if not 1 <= args.max_trajectories <= 128 or not 1 <= args.max_context_tokens <= 4096:
        raise ValueError('Bounded qualification fixture/context limits required')
    source, output = remote(args.selection), remote(args.output)
    source_proof = fingerprints(source)
    if source_proof['sha256'] != args.selection_sha256:
        raise ValueError('Selection JSON SHA differs')
    selection = read(source)
    if selection['model_revision'] != REVISION:
        raise ValueError('Selection model revision differs')
    output.mkdir(parents=True, exist_ok=False)
    lock = (output/'capture.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(output/'intent.json', dict(status='BOUNDED_FROZEN_HF_REPLAY_INTENT', source=source_proof,
        max_trajectories=args.max_trajectories, max_context_tokens=args.max_context_tokens,
        job_id=os.environ['SLURM_JOB_ID'], actor_optimizer_updates=0, epoch_time=time.time()))
    selected, prefix_proofs = load_selected(selection, args.max_trajectories, args.max_context_tokens)
    model_proofs, model_config = verify_model(args.model, selection)
    native_manifests, native_proofs = verify_native_sources(selection)
    if any(token >= model_config['vocab_size'] for row in selected for segment in row['segments']
           for token in segment['context_tokens']):
        raise ValueError('Saved actual token ID exceeds official vocabulary')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['QWEN7B_QUESTION_MAX_TOKENS'] = str(QMAX)
    os.environ['SEARCH_G1_B_PREFIX_TOKENS'] = str(PREFIX)
    from .packed_features import FeatureReader
    native_reader = FeatureReader(native_manifests)
    for row in selected:
        if row['source_native_manifest'] is not None and str(remote(row['source_native_manifest'])) not in native_manifests:
            raise ValueError('Reference manifest is not the original sealed prefix feature locator')
        if row['trajectory_id'] not in native_reader.records:
            raise ValueError('Native reference lacks a declared trajectory')
        native = native_reader.records[row['trajectory_id']][1]
        if str(native['qid']) != row['qid'] or native['question_length'] != row['question_length'] or native['policy_length'] != row['policy_length']:
            raise ValueError('Native reference question/trajectory geometry differs')
    import transformers
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(args.model), torch_dtype=torch.bfloat16,
        attn_implementation=args.attention_implementation, local_files_only=True,
        trust_remote_code=False, low_cpu_mem_usage=True).eval().requires_grad_(False).to(device)
    if len(model.layers) != 28 or model.config.hidden_size != DIM or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen official7B model geometry/gradient gate failed')
    passes = 0
    def capture(context, positions):
        nonlocal passes
        if positions != list(range(len(context))):
            raise ValueError('Actual causal segment positions changed')
        ids = torch.tensor([context], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        rope = torch.tensor([positions], dtype=torch.long, device=device)
        with torch.inference_mode():
            value = model(input_ids=ids, attention_mask=mask, position_ids=rope,
                          use_cache=False, output_hidden_states=True, return_dict=True)
            if len(value.hidden_states) != 29:
                raise ValueError('Unexpected HF hidden_states residual boundary')
            result = {layer:value.hidden_states[layer+1][0].detach().to('cpu', dtype=torch.bfloat16, copy=True)
                      for layer in LAYERS}
        passes += 1
        return result
    writers = {layer:ReplayLayerWriter(output/('layer-%02d'%layer), layer, selected) for layer in LAYERS}
    comparisons = {kind:Comparison() for kind in ('all_retained', 'initial_question', 'policy_prefix20')}
    started = time.monotonic()
    try:
        for index, row in enumerate(selected):
            replay = replay_features(row, capture)
            native = native_reader.get([row['trajectory_id']], 'cpu')
            own = replay[26]
            if (not torch.equal(own['attention_mask'], native['attention_mask'][0]) or
                    not torch.equal(own['position_ids'], native['position_ids'][0])):
                raise ValueError('Native and replay retained slots/positions differ')
            qlen, plen = row['question_length'], row['policy_length']
            comparisons['all_retained'].add(own['features'][own['attention_mask']], native['features'][0][own['attention_mask']])
            comparisons['initial_question'].add(own['features'][:qlen], native['features'][0, :qlen])
            comparisons['policy_prefix20'].add(own['features'][QMAX:QMAX+plen], native['features'][0, QMAX:QMAX+plen])
            for layer in LAYERS:
                writers[layer].write(row, replay[layer])
            atomic(output/'progress.json', dict(status='ACTUAL_FROZEN_CAUSAL_RECAPTURE_ADVANCING',
                trajectories=index+1, expected_trajectories=len(selected), causal_prefill_passes=passes,
                retained_policy_tokens=sum(r['policy_length'] for r in selected[:index+1]),
                elapsed_seconds=time.monotonic()-started, actor_optimizer_updates=0, epoch_time=time.time()))
        manifests = {str(layer):writers[layer].finish() for layer in LAYERS}
    finally:
        native_reader.close()
    comparison = {kind:metric.result() for kind, metric in comparisons.items()}
    atomic(output/'native-layer26-comparison.json', dict(metrics=comparison,
        reference='vllm_actual_native_sampling_postblock26_hidden_plus_residual',
        replay='huggingface_causal_prefill_hidden_states27',
        bit_exact_claimed=False, accepted_for_training=False, numeric_acceptance_pending=True))
    proof = dict(status='BOUNDED_HF_REPLAY_CAPTURE_AND_NATIVE_COMPARISON_COMPLETE',
        model=MODEL, model_revision=REVISION, capture_layers=list(LAYERS),
        hf_hidden_state_indices=[layer+1 for layer in LAYERS], capture_kind=CAPTURE_KIND,
        trajectories=len(selected), question_max_tokens=QMAX, prefix_tokens=PREFIX,
        question_source='first_turn0_initial_prompt_slots_only_causal_attention',
        policy_source='own_saved_actual_request_and_retained_segment_tokens_only',
        future_generated_tokens_used=False, actor_frozen=True, actor_optimizer_updates=0,
        causal_prefill_passes=passes, expected_causal_prefill_passes=sum(len(r['segments']) for r in selected),
        source_selection=source_proof, source_prefix_files=prefix_proofs,
        model_files=model_proofs, source_native_files=native_proofs, feature_manifests=manifests,
        comparison_file=fingerprints(output/'native-layer26-comparison.json'),
        code=fingerprints(__file__),
        code_files={name:fingerprints(Path(__file__).with_name(name)) for name in
                    ('replay_probe_layers.py', 'packed_features.py', 'native_capture.py', 'download_model.py')},
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        attention_implementation=args.attention_implementation, device=str(device),
        gpu_name=torch.cuda.get_device_name(device), job_id=os.environ['SLURM_JOB_ID'],
        bit_exact_native_vllm_claimed=False, accepted_for_training=False,
        full_bank_recapture_claimed=False, epoch_time=time.time())
    if proof['causal_prefill_passes'] != proof['expected_causal_prefill_passes']:
        raise ValueError('Not exactly one causal frozen forward per saved segment')
    atomic(output/'complete.json', proof)
    print(json.dumps(dict(status=proof['status'], output=str(output), trajectories=len(selected),
                          causal_prefill_passes=passes, native_comparison=comparison,
                          accepted_for_training=False)), flush=True)
    lock.close()


if __name__ == '__main__':
    main()
