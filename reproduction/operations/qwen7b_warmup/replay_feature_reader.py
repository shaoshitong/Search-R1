"""Strict reader for independently SHA-bound frozen-HF replay layer banks.

The old native-only FeatureReader is deliberately unchanged.  This reader only
accepts ReplayLayerWriter's LOSSLESS_HF_REPLAY_FEATURES_COMPLETE format, checks
every manifest/file SHA and every reconstructed retained feature SHA, and emits
explicit capture metadata for NativeBlockProbe.  Reading does not accept the
HF-vLLM backend change for training.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import zlib

from .replay_probe_layers import CAPTURE_KIND, DIM, LAYERS, MODEL, PREFIX, QMAX, REVISION, WIDTH, fingerprints, remote


FILES = {'question.bin', 'delta.bin', 'policy.bin', 'index.jsonl', 'prototype.bin'}


def hexadecimal(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


class ReplayFeatureReader:
    """``get(ids, device)`` matches the existing feature-reader tensor interface.

    Supply manifest descriptors {path,sha256,bytes}, not unbound path strings.
    ``synthetic_fixture`` only enables tiny local unit-test banks; production
    defaults require remote paths and hidden3584.  No test bank is an accepted
    model/feature asset.
    """
    def __init__(self, manifests, capture_layer, synthetic_fixture=False, fixture_dimension=None):
        if capture_layer not in LAYERS:
            raise ValueError('Replay capture layer must be20,23,or26')
        if fixture_dimension is not None and not synthetic_fixture:
            raise ValueError('Synthetic dimension override is restricted to explicit unit fixtures')
        self.layer = capture_layer
        self.dimension = fixture_dimension if synthetic_fixture and fixture_dimension is not None else DIM
        if not integer(self.dimension, 1):
            raise ValueError('Positive feature dimension required')
        self.records, self.cache, self.prototypes = {}, OrderedDict(), {}
        self.manifests, self.files = [], []
        self.synthetic_fixture = synthetic_fixture
        self.closed = False
        questions = {}
        if not manifests:
            raise ValueError('At least one SHA-bound replay manifest is required')
        for descriptor in manifests:
            if not isinstance(descriptor, dict) or not hexadecimal(descriptor.get('sha256')):
                raise ValueError('Every replay manifest requires an explicit SHA descriptor')
            path = self.path(descriptor['path'])
            actual = fingerprints(path)
            if actual['sha256'] != descriptor['sha256'] or actual['bytes'] != descriptor['bytes']:
                raise ValueError('Replay manifest SHA/bytes differ')
            manifest = json.loads(path.read_text())
            if (manifest['status'] != 'LOSSLESS_HF_REPLAY_FEATURES_COMPLETE' or
                    manifest['source_runtime'] != 'huggingface_frozen_causal_prefill' or
                    manifest['layout'] != 'lossless_question_base_xor_policy_v1'):
                raise ValueError('Only the declared frozen-HF replay writer format is accepted')
            if (manifest['capture_layer'] != capture_layer or manifest['feature_layer'] != capture_layer or
                    manifest['capture_kind'] != CAPTURE_KIND or
                    manifest['dimension'] != self.dimension or manifest['question_max_tokens'] != QMAX or
                    manifest['prefix_tokens'] != PREFIX):
                raise ValueError('Replay layer/boundary/geometry differs from the requested probe')
            if manifest['model'] != MODEL or manifest['model_revision'] != REVISION:
                raise ValueError('Replay model revision differs')
            if manifest['actor_optimizer_updates'] != 0 or manifest['bit_exact_native_vllm_claimed'] is not False:
                raise ValueError('Frozen replay provenance/bit-exact claim differs')
            # Initial released writer manifests bind dtype through this exact
            # versioned layout and source runtime; newer explicit dtype fields
            # must agree.  This is not inference from the payload's appearance.
            if manifest.get('feature_dtype', 'bfloat16_bits_uint16') != 'bfloat16_bits_uint16':
                raise ValueError('Replay payload is not the versioned BF16-bit layout')
            if set(manifest['files']) != FILES or not integer(manifest['count'], 1):
                raise ValueError('Unexpected or incomplete replay packed-file set')
            for name, item in manifest['files'].items():
                source = self.path(item['path'])
                if source != path.parent/name:
                    raise ValueError('Replay file locator escaped its manifest directory')
                proof = fingerprints(source)
                if proof['sha256'] != item['sha256'] or proof['bytes'] != item['bytes']:
                    raise ValueError('Replay source file SHA/bytes differ: '+name)
                self.files.append(proof)
            if manifest['files']['prototype.bin']['bytes'] != QMAX*self.dimension*2:
                raise ValueError('Prototype shape/uint16 dtype size differs')
            count = 0
            with (path.parent/'index.jsonl').open() as stream:
                for line in stream:
                    row = json.loads(line)
                    self.validate_row(row, manifest)
                    tid = row['trajectory_id']
                    if tid in self.records:
                        raise ValueError('Duplicate replay trajectory identity')
                    qid = str(row['qid'])
                    binding = (row['question_length'], row['prompt_sha256'])
                    if questions.setdefault(qid, binding) != binding:
                        raise ValueError('Question identity/prompt changed across replay trajectories')
                    self.records[tid] = (str(path.parent), row)
                    count += 1
            if count != manifest['count']:
                raise ValueError('Replay index count differs from sealed manifest')
            self.manifests.append(actual)
        self.proof = dict(status='HF_REPLAY_READER_SHA_GEOMETRY_AND_INDEX_VERIFIED',
            capture_layer=self.layer, capture_kind=CAPTURE_KIND, model_revision=REVISION,
            count=len(self.records), dimension=self.dimension, dtype='bfloat16_bits_uint16',
            dtype_binding='versioned_writer_layout_and_source_runtime', manifests=self.manifests,
            files=self.files, synthetic_fixture=synthetic_fixture, accepted_for_training=False)

    def path(self, path):
        value = Path(path)
        if self.synthetic_fixture:
            if not value.is_absolute() or value.resolve() != value:
                raise ValueError('Absolute nonsymlink synthetic fixture path required')
            return value
        return remote(value)

    def validate_row(self, row, manifest):
        tid, qid = row['trajectory_id'], row['qid']
        if not isinstance(tid, str) or not tid or type(qid) not in (str, int) or not str(qid):
            raise ValueError('Real replay trajectory/question identities required')
        qlen, plen = row['question_length'], row['policy_length']
        if not integer(qlen, 1) or qlen > QMAX or not integer(plen) or plen > PREFIX:
            raise ValueError('Question or retained-prefix width differs')
        if (row['dimension'] != self.dimension or row['capture_layer'] != self.layer or
                row['capture_kind'] != CAPTURE_KIND):
            raise ValueError('Row layer/geometry differs from manifest')
        positions = row['policy_positions']
        if (not isinstance(positions, list) or len(positions) != plen or
                any(not integer(position) for position in positions) or
                (plen and (positions[0] != qlen or any(b <= a for a,b in zip(positions, positions[1:]))))):
            raise ValueError('Retained policy positions do not preserve the actual causal order')
        for key in ('prompt_sha256', 'policy_token_sha256', 'source_prefix_sha256', 'retained_feature_sha256'):
            if not hexadecimal(row[key]):
                raise ValueError('Missing own-trajectory source/content SHA: '+key)
        for offset, length, filename in (
                ('question_offset', 'question_bytes', 'question.bin'),
                ('delta_offset', 'delta_bytes', 'delta.bin')):
            if (not integer(row[offset]) or not integer(row[length]) or
                    row[offset]+row[length] > manifest['files'][filename]['bytes']):
                raise ValueError('Replay compressed span is outside its verified file')
        if row['question_bytes'] == 0:
            raise ValueError('A real question needs a compressed source span')
        offset = row['policy_offset']
        if (not integer(offset) or offset % (self.dimension*2) or
                offset+plen*self.dimension*2 > manifest['files']['policy.bin']['bytes']):
            raise ValueError('Policy BF16 token-vector span is misaligned/outside its verified file')

    def handles(self, root):
        import numpy as np
        if self.closed:
            raise ValueError('Replay reader is closed')
        if root not in self.cache:
            path = Path(root)
            if root not in self.prototypes:
                self.prototypes[root] = np.frombuffer((path/'prototype.bin').read_bytes(), dtype=np.uint16).reshape(QMAX, self.dimension)
            self.cache[root] = {name:(path/name).open('rb') for name in ('question.bin', 'delta.bin', 'policy.bin')}
            if len(self.cache) > 16:
                _, old = self.cache.popitem(last=False)
                for stream in old.values():
                    stream.close()
        self.cache.move_to_end(root)
        return self.cache[root], self.prototypes[root]

    @staticmethod
    def span(stream, offset, size):
        stream.seek(offset)
        payload = stream.read(size)
        if len(payload) != size:
            raise ValueError('Replay packed file became truncated after validation')
        return payload

    def get(self, ids, device):
        import numpy as np
        import torch
        ids = list(ids)
        if not ids or any(tid not in self.records for tid in ids):
            raise ValueError('Every requested replay trajectory must exist in the sealed bank')
        features = torch.zeros((len(ids), WIDTH, self.dimension), dtype=torch.bfloat16)
        masks = torch.zeros((len(ids), WIDTH), dtype=torch.bool)
        positions = torch.zeros((len(ids), WIDTH), dtype=torch.long)
        qids = []
        for i, tid in enumerate(ids):
            root, row = self.records[tid]
            streams, prototype = self.handles(root)
            qlen, plen = row['question_length'], row['policy_length']
            payload = self.span(streams['question.bin'], row['question_offset'], row['question_bytes'])
            question_delta = zlib.decompress(payload)
            if len(question_delta) != qlen*self.dimension*2:
                raise ValueError('Question decompressed BF16-bit shape differs')
            question = np.bitwise_xor(prototype[:qlen],
                np.frombuffer(question_delta, dtype=np.uint16).reshape(qlen, self.dimension))
            if row['delta_bytes']:
                payload = self.span(streams['delta.bin'], row['delta_offset'], row['delta_bytes'])
                own_delta = zlib.decompress(payload)
                if len(own_delta) != qlen*self.dimension*2:
                    raise ValueError('Own-trajectory question delta shape differs')
                question = np.bitwise_xor(question,
                    np.frombuffer(own_delta, dtype=np.uint16).reshape(qlen, self.dimension))
            policy_payload = self.span(streams['policy.bin'], row['policy_offset'], plen*self.dimension*2)
            if hashlib.sha256(question.tobytes()+policy_payload).hexdigest() != row['retained_feature_sha256']:
                raise ValueError('Reconstructed own retained feature SHA differs')
            policy = np.frombuffer(policy_payload, dtype=np.uint16).reshape(plen, self.dimension)
            features[i, :qlen] = torch.from_numpy(question.copy()).view(torch.bfloat16)
            features[i, QMAX:QMAX+plen] = torch.from_numpy(policy.copy()).view(torch.bfloat16)
            masks[i, :qlen] = True
            masks[i, QMAX:QMAX+plen] = True
            positions[i, :qlen] = torch.arange(qlen)
            positions[i, QMAX:QMAX+plen] = torch.tensor(row['policy_positions'], dtype=torch.long)
            if not bool(torch.isfinite(features[i][masks[i]]).all()):
                raise ValueError('Nonfinite reconstructed real BF16 feature')
            qids.append(str(row['qid']))
        return dict(features=features.to(device), attention_mask=masks.to(device),
                    position_ids=positions.to(device), capture_layer=self.layer,
                    capture_kind=CAPTURE_KIND, model_revision=REVISION,
                    trajectory_ids=ids, qids=qids)

    def close(self):
        for own in self.cache.values():
            for stream in own.values():
                stream.close()
        self.cache.clear()
        self.prototypes.clear()
        self.closed = True
