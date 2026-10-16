"""Bounded CPU-only cost aggregation from committed native trajectory shards.

The training checkout is never edited. Output is small scalar evidence under
the current run's preparation directory; raw trajectories remain remote.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.hash.update(value)
        self.bytes += len(value)
        return value


def source_base(path, run_id):
    path = Path(path)
    candidates = [p for p in path.parents if p.parent == Path('/mnt/data/shaoshitong')]
    if (len(candidates) != 1 or path.resolve() != path
            or not candidates[0].name.startswith('search-g1-prefix-diversity200-')
            or read(candidates[0]/'preparation/launch-config.json')['run_id'] != run_id):
        raise ValueError('Shard is outside canonical namespaces of this logical run')
    return candidates[0]


def rows(item, run_id):
    path = Path(item['path'])
    source_base(path, run_id)
    if path.stat().st_size != item['bytes']:
        raise ValueError('Committed shard byte count differs')
    count = 0
    with path.open('rb') as stream:
        hashed = HashReader(stream)
        with gzip.GzipFile(fileobj=hashed, mode='rb') as decoded:
            for line in decoded:
                count += 1
                yield json.loads(line)
        if hashed.bytes != item['bytes'] or hashed.hash.hexdigest() != item['sha256']:
            raise ValueError('Streamed committed shard SHA256 differs')
    if count != item['count']:
        raise ValueError('Committed trajectory count differs')


def summarize(base, outer, run_id):
    root = base/'outers'/('outer-%06d' % outer)
    paths = [root/'complete.json', root/'prefix/complete.json', root/'full/complete.json']
    joint, prefix, full = map(read, paths)
    for receipt in (joint, prefix, full):
        if receipt['run_id'] != run_id or receipt['outer'] != outer:
            raise ValueError('Committed generation identity differs')
    if (prefix['trajectories'] != 25600 or full['trajectories'] != 5120
            or prefix['actor_version'] != full['actor_version']):
        raise ValueError('Original generation budgets or actor identity differ')
    original = {}
    counts = {}
    for item in prefix['state_shards']:
        for row in rows(item, run_id):
            tid = row['trajectory_id']
            calls = row['retrieval_calls']
            if (row['run_id'] != run_id or row['outer'] != outer
                    or row['actor_version'] != prefix['actor_version']
                    or tid in original or type(calls) is not int or calls < 0):
                raise ValueError('Prefix identity/count differs')
            qid = row['group']['qid']
            original[tid] = (calls, qid, row['response_index'], row['prefix_sha256'])
            counts.setdefault(qid, set()).add(row['response_index'])
    if len(original) != 25600 or len(counts) != 5120 or any(v != set(range(5)) for v in counts.values()):
        raise ValueError('Prefix groups are not all original G5')
    seen = set()
    continued_groups = {}
    inherited_calls = total_full_calls = 0
    for item in full['shards']:
        for row in rows(item, run_id):
            tid = row['trajectory_id']
            calls = row['retrieval_calls']
            if (row['run_id'] != run_id or row['outer'] != outer
                    or row['actor_version'] != full['actor_version']
                    or tid not in original or tid in seen or type(calls) is not int):
                raise ValueError('Continuation identity/count differs')
            inherited, qid, index, prefix_sha = original[tid]
            if (row['qid'] != qid or row['response_index'] != index
                    or row['prefix_sha256'] != prefix_sha or calls < inherited):
                raise ValueError('Continuation lost original prefix or retrieval counter')
            seen.add(tid)
            continued_groups.setdefault(qid, set()).add(index)
            inherited_calls += inherited
            total_full_calls += calls
    if len(seen) != 5120 or len(continued_groups) != 1024 or any(v != set(range(5)) for v in continued_groups.values()):
        raise ValueError('Continuation groups are not all selected original G5')
    all_prefix_calls = sum(v[0] for v in original.values())
    return dict(schema_version=1, run_id=run_id, outer=outer, execution_base=str(base),
        observed_unix=time.time(), prefix_all25600_logical_retrieval_calls=all_prefix_calls,
        selected5120_inherited_prefix_retrieval_calls=inherited_calls,
        selected5120_full_trajectory_retrieval_calls=total_full_calls,
        continuation_new_logical_retrieval_calls=total_full_calls-inherited_calls,
        total_distinct_logical_retrieval_calls=all_prefix_calls+total_full_calls-inherited_calls,
        actual_prefix_trajectories=25600, actual_continued_trajectories=5120,
        receipt_sha256={str(p):digest(p) for p in paths},
        source_namespace_config_sha256={str(p):digest(p) for p in {
            source_base(item['path'], run_id)/'preparation/launch-config.json'
            for item in prefix['state_shards']+full['shards']}},
        shard_sha256_verified_in_same_stream=True,
        scope='committed outer logical search actions only; HTTP retries, cache hits, failed attempts and QA are separate',
        actor512_is_subset_not_additive=True, raw_trajectory_output_returned=False,
        model_tensors_read=False, new_model_forward=False, training_modified=False)


def collect(bases, run_id, maximum_new_outers=2):
    if not 1 <= maximum_new_outers <= 4:
        raise ValueError('Bound each invocation to at most four new complete outers')
    canonical = []
    complete = {}
    for name in bases:
        base = Path(name)
        if (base.parent != Path('/mnt/data/shaoshitong') or base.resolve() != base
                or not base.name.startswith('search-g1-prefix-diversity200-')):
            raise ValueError('Only this experiment remote namespace is allowed')
        c = read(base/'preparation/launch-config.json')
        if c['run_id'] != run_id:
            raise ValueError('Foreign logical run namespace')
        canonical.append(base)
        for path in sorted((base/'outers').glob('outer-*/complete.json')):
            joint = read(path)
            outer = joint['outer']
            if joint['run_id'] != run_id or outer in complete:
                raise ValueError('Foreign or duplicated complete outer')
            complete[outer] = (base, path)
    output = canonical[-1]/'preparation/retrieval-cost-v1'
    output.mkdir(exist_ok=True)
    proofs = {}
    created = []
    for outer, (base, path) in sorted(complete.items()):
        destination = output/('outer-%06d.json' % outer)
        if destination.exists():
            proof = read(destination)
            if (proof['run_id'] != run_id or proof['outer'] != outer
                    or proof['execution_base'] != str(base)
                    or any(digest(p) != expected for p, expected in
                        dict(proof['receipt_sha256'], **proof['source_namespace_config_sha256']).items())):
                raise ValueError('Existing cost proof no longer binds the original receipts')
            proofs[str(outer)] = proof
        elif len(created) < maximum_new_outers:
            proof = summarize(base, outer, run_id)
            descriptor, staging = tempfile.mkstemp(prefix='.cost-proof-', dir=output)
            try:
                with os.fdopen(descriptor, 'w') as stream:
                    json.dump(proof, stream, sort_keys=True, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(staging, destination)
            finally:
                os.unlink(staging)
            proofs[str(outer)] = proof
            created.append(outer)
    return dict(run_id=run_id, observed_unix=time.time(), output_directory=str(output),
        new_complete_outer_proofs=created, complete_outer_rows=proofs,
        pending_complete_outers=sorted(set(complete)-set(map(int, proofs))),
        total_distinct_logical_retrieval_calls=sum(p['total_distinct_logical_retrieval_calls'] for p in proofs.values()),
        model_tensors_read=False, new_model_forward=False, training_modified=False)
