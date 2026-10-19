"""Lossless native BF16 features: one question base plus per-trajectory XOR.

Question differences between native batching kernels are NOT assumed zero.
Every differing bit is retained in a compressed XOR. Each trajectory's own
policy features remain raw; exact reconstruction is checked by a byte hash.
"""
import hashlib
import json
import os
from pathlib import Path
import zlib
from collections import OrderedDict
from .native_capture import DIM, QMAX, PREFIX, WIDTH
from .download_model import atomic


class NativeFeatureWriter:
    def __init__(self, directory, rank, trajectory_ids, selected_capacity=None):
        assert selected_capacity is None
        self.root = Path(directory) / ('rank-%d-packed' % rank)
        self.root.mkdir(exist_ok=False)
        self.allowed = set(trajectory_ids)
        assert len(self.allowed) == len(trajectory_ids)
        self.done, self.questions = set(), {}
        self.prototype = None
        self.streams = {name: (self.root / name).open('xb') for name in ('question.bin', 'delta.bin', 'policy.bin', 'index.jsonl')}
        self.hashes = {name: hashlib.sha256() for name in self.streams}

    def put(self, name, data):
        handle = self.streams[name]
        offset = handle.tell()
        handle.write(data)
        self.hashes[name].update(data)
        return offset

    def write(self, prefix, data):
        import numpy as np
        import torch
        tid, qid = prefix['trajectory_id'], str(prefix['qid'])
        assert tid in self.allowed and tid not in self.done
        qlen, plen = data['question_length'], data['policy_length']
        array = data['features'].contiguous().view(torch.uint16).numpy()
        q = array[:qlen].copy()
        if self.prototype is None:
            self.prototype = np.zeros((QMAX,DIM), dtype=np.uint16)
            self.prototype[:qlen] = q
            with (self.root/'prototype.bin').open('xb') as handle:
                handle.write(self.prototype.tobytes());handle.flush();os.fsync(handle.fileno())
        if qid not in self.questions:
            compressed = zlib.compress(np.bitwise_xor(q,self.prototype[:qlen]).tobytes(),1)
            offset = self.put('question.bin', compressed)
            self.questions[qid] = dict(array=q, offset=offset, bytes=len(compressed), count=0, qlen=qlen,
                prompt_sha256=hashlib.sha256(json.dumps(prefix['initial_prompt_token_ids']).encode()).hexdigest())
        base = self.questions[qid]
        assert base['qlen'] == qlen and base['prompt_sha256'] == hashlib.sha256(json.dumps(prefix['initial_prompt_token_ids']).encode()).hexdigest()
        xor = np.bitwise_xor(q, base['array'])
        delta = zlib.compress(xor.tobytes(), 1) if np.any(xor) else b''
        delta_offset = self.put('delta.bin', delta)
        policy = array[QMAX:QMAX+plen].copy()
        policy_offset = self.put('policy.bin', policy.tobytes())
        # Check lossless reconstruction now, before publishing a feature row.
        restored = base['array'] if not delta else np.bitwise_xor(base['array'], np.frombuffer(zlib.decompress(delta), dtype=np.uint16).reshape(q.shape))
        assert np.array_equal(restored, q)
        expected = hashlib.sha256(q.tobytes() + policy.tobytes()).hexdigest()
        row = dict(trajectory_id=tid, qid=qid, question_length=qlen, policy_length=plen,
            question_offset=base['offset'], question_bytes=base['bytes'], delta_offset=delta_offset, delta_bytes=len(delta),
            policy_offset=policy_offset, dimension=DIM, policy_positions=data['position_ids'][QMAX:QMAX+plen].tolist(),
            retained_feature_sha256=expected)
        self.put('index.jsonl', (json.dumps(row, separators=(',', ':'))+'\n').encode())
        self.done.add(tid)
        base['count'] += 1
        if base['count'] == 5:
            del self.questions[qid]
        return dict(feature_rank=None, feature_row=len(self.done)-1, feature_file=str(self.root/'complete.json'),
            feature_layout='lossless_question_base_xor_policy_v1', feature_actor_version=prefix['actor_version'],
            feature_layer=26, feature_boundary='post_block_hidden_plus_residual',
            feature_dtype='bfloat16_bits_uint16', future_information_included=False)

    def finish(self):
        assert self.done == self.allowed and not self.questions
        files = {}
        for name, handle in self.streams.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            path = self.root/name
            files[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=self.hashes[name].hexdigest())
        prototype = self.root/'prototype.bin'
        files['prototype.bin'] = dict(path=str(prototype),bytes=prototype.stat().st_size,sha256=hashlib.sha256(prototype.read_bytes()).hexdigest())
        path = self.root/'complete.json'
        atomic(path, dict(status='LOSSLESS_NATIVE_FEATURES_COMPLETE', count=len(self.done), files=files,
            dimension=DIM, question_max_tokens=QMAX, prefix_tokens=PREFIX, feature_layer=26))
        return path


class FeatureReader:
    def __init__(self, manifests):
        self.records, self.cache = {}, OrderedDict()
        for manifest in manifests:
            proof = json.loads(Path(manifest).read_text())
            assert proof['status'] == 'LOSSLESS_NATIVE_FEATURES_COMPLETE'
            root = Path(manifest).parent
            for line in (root/'index.jsonl').open():
                row = json.loads(line)
                assert row['trajectory_id'] not in self.records
                self.records[row['trajectory_id']] = (str(root), row)

    def close(self):
        for handles,_ in self.cache.values():
            for handle in handles.values():handle.close()
        self.cache.clear()

    def handles(self, root):
        import numpy as np
        if root not in self.cache:
            base = Path(root)
            self.cache[root] = ({name:(base/name).open('rb') for name in ('question.bin','delta.bin','policy.bin')},
                np.frombuffer((base/'prototype.bin').read_bytes(),dtype=np.uint16).reshape(QMAX,DIM))
            if len(self.cache) > 16:
                _, (handles, _) = self.cache.popitem(last=False)
                for handle in handles.values():handle.close()
        self.cache.move_to_end(root)
        return self.cache[root]

    def get(self, ids, device):
        import numpy as np
        import torch
        features = torch.zeros((len(ids), WIDTH, DIM), dtype=torch.bfloat16)
        mask = torch.zeros((len(ids), WIDTH), dtype=torch.bool)
        positions = torch.zeros((len(ids), WIDTH), dtype=torch.long)
        for i, tid in enumerate(ids):
            root, row = self.records[tid]
            handles, prototype = self.handles(root)
            qlen, plen = row['question_length'], row['policy_length']
            handle = handles['question.bin']; handle.seek(row['question_offset'])
            delta = np.frombuffer(zlib.decompress(handle.read(row['question_bytes'])),dtype=np.uint16).reshape(qlen,DIM)
            q = np.bitwise_xor(prototype[:qlen],delta)
            if row['delta_bytes']:
                handle = handles['delta.bin']; handle.seek(row['delta_offset'])
                delta = np.frombuffer(zlib.decompress(handle.read(row['delta_bytes'])), dtype=np.uint16).reshape(q.shape)
                np.bitwise_xor(q, delta, out=q)
            handle = handles['policy.bin']; handle.seek(row['policy_offset'])
            p = np.frombuffer(handle.read(plen*DIM*2), dtype=np.uint16).reshape(plen,DIM).copy()
            assert hashlib.sha256(q.tobytes()+p.tobytes()).hexdigest() == row['retained_feature_sha256']
            features[i,:qlen] = torch.from_numpy(q).view(torch.bfloat16)
            features[i,QMAX:QMAX+plen] = torch.from_numpy(p).view(torch.bfloat16)
            mask[i,:qlen] = True; mask[i,QMAX:QMAX+plen] = True
            positions[i,:qlen] = torch.arange(qlen)
            positions[i,QMAX:QMAX+plen] = torch.tensor(row['policy_positions'],dtype=torch.long)
        return dict(features=features.to(device), attention_mask=mask.to(device), position_ids=positions.to(device))
