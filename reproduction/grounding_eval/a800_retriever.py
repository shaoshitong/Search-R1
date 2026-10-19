"""Task-owned exact-spec A800 retriever with admission and pre-listen parity.

The historical DenseRetriever and batching implementation are read-only imports.
No HTTP listener becomes available until sixteen real pilot searches reproduce
all post-truncation observation token IDs. Holds auxiliary UUID OFD leases.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import time
import uuid

SOURCE_SHA = {
    'reproduction/prefix20_online/retriever.py': 'e7abe00c2e996bdc0d9e4fd92dc62c8fc6c0d51c71e45a5de1858edbfd82dbf9',
    'search_r1/search/retrieval_server.py': 'b0e70e06bbd7d4d820ff7c550142fa6b22725789711f927bdc4960dcdc766ca1',
    'search_r1/llm_agent/generation.py': '38785f5f611f35fbaf8911c21c0434d915a83498d6d705c01fe27946129afc4f',
    'reproduction/common8/temporary_cache.py': '94087c2caea98e81c836d179be79d87f6f164f3adfbd6da7931ae90098de55cd',
}


def single_gpu_entrypoint(source):
    """Adapt only the historical launch guards, never the retrieval computation.

    FAISS may return a plain GpuIndex instead of IndexShards for one GPU. The
    assigned visible-device count, corpus dimensions and real observation-token
    parity remain mandatory. Exact text guards fail closed if upstream changes.
    """
    edits = {
        'choices=(2,8)': 'choices=(1,2,8)',
        'retriever.index.count() != args.gpu_shards':
            "(retriever.index.count() if hasattr(retriever.index, 'count') else 1) != args.gpu_shards",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError('Unexpected historical single-GPU launch guard')
        source = source.replace(old, new)
    compile(source, '<grounding-single-gpu-main>', 'exec')
    return source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    p = Path(path); temp = p.with_name(p.name + '.partial-' + uuid.uuid4().hex)
    with temp.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, p)


def lease_aux(uuids, port):
    rows = list(csv.reader(io.StringIO(subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True))))
    processes = list(csv.reader(io.StringIO(subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True))))
    held, indices = [], []
    try:
        for gpu in uuids:
            if not re.fullmatch(r'GPU-[0-9a-f-]+', gpu):
                raise ValueError('Bad UUID')
            lock = Path('/data/shaoshitong/.search-g1-gpu-uuid-locks') / (gpu + '.lock')
            fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            held.append(fd)
            fcntl.fcntl(fd, fcntl.F_OFD_SETLK,
                        struct.pack('hhqqi4x', fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
            match = [r for r in rows if r[1].strip() == gpu]
            if len(match) != 1 or int(match[0][2]) > 512 or int(match[0][3]) != 0:
                raise RuntimeError('Auxiliary GPU is not idle')
            if any(r and r[0].strip() == gpu for r in processes):
                raise RuntimeError('Auxiliary GPU has an existing process')
            indices.append(int(match[0][0]))
        # A800 is a standalone SSH host, not a scheduler allocation. Fail closed
        # if a Slurm agent appears, rather than infer reservation from idle GPUs.
        if any(p.read_text().strip() == 'slurmd' for p in Path('/proc').glob('*/comm')
               if p.is_file()):
            raise RuntimeError('Unexpected Slurm agent requires scheduler admission')
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) == os.getpid():
                continue
            try:
                args = (p / 'cmdline').read_bytes().split(b'\0')
                if any(x.startswith(b'reproduction.prefix20_online.') for x in args):
                    raise RuntimeError('A prefix training/bootstrap process is present')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
        return held, indices, rows
    except BaseException:
        for fd in held:
            os.close(fd)
        raise


def verify_parity(retriever, *, records, tokenizer_path, receipt):
    from verl.utils import hf_tokenizer
    from search_r1.llm_agent.generation import LLMGenerationManager
    from types import SimpleNamespace
    tok = hf_tokenizer(tokenizer_path)
    manager = LLMGenerationManager.__new__(LLMGenerationManager)
    manager.tokenizer, manager.config = tok, SimpleNamespace(max_obs_length=500)
    selected = []
    for p in sorted(Path(records).glob('*.json')):
        row = json.loads(p.read_text())
        # Each actual retrieved observation must correspond to its exact model
        # action; raw output includes discarded tails, so use proven kept IDs.
        for turn in row['behavior_turns']:
            if not turn['observation_token_ids']:
                continue
            text = tok.decode(turn['raw_generated_token_ids'][:turn['kept_token_count']],
                              skip_special_tokens=True, clean_up_tokenization_spaces=False)
            actions, contents = manager.postprocess_predictions([text])
            if actions != ['search']:
                raise ValueError('Stored observation has no strict search action')
            selected.append((row, turn, contents[0], sha(p)))
            if len(selected) == 16:
                break
        if len(selected) == 16:
            break
    if len(selected) != 16:
        raise ValueError('Sixteen real normal searches are required')
    queries = [v[2] for v in selected]
    docs, scores = retriever.batch_search(query_list=queries, num=3, return_score=True)
    if len(docs) != 16 or len(scores) != 16 or any(
            len(d) != 3 or len(s) != 3 for d, s in zip(docs, scores)):
        raise ValueError('Parity retrieval returned incomplete or non-top3 batch')
    observations = []
    for items, values in zip(docs, scores):
        payload = [{'document': d, 'score': float(s)} for d, s in zip(items, values)]
        body = manager._passages2string(payload)
        observations.append('\n\n<information>' + body.strip() + '</information>\n\n')
    actual = manager._process_next_obs(observations).detach().cpu().tolist()
    if len(actual) != 16:
        raise ValueError('Parity tokenization returned incomplete batch')
    evidence = []
    for (row, turn, query, record_sha), ids in zip(selected, actual):
        ids = [v for v in ids if v != tok.pad_token_id]
        expected = turn['observation_token_ids']
        digest = lambda v: hashlib.sha256(json.dumps(v, separators=(',', ':')).encode()).hexdigest()
        evidence.append(dict(qid=row['qid'], evaluation_ordinal=row['evaluation_ordinal'],
            turn_index=turn['turn_index'], query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            source_record_sha256=record_sha, expected_token_sha256=digest(expected),
            actual_token_sha256=digest(ids), tokens=len(ids), exact_equal=ids == expected))
    if len(evidence) != 16:
        raise ValueError('Incomplete parity evidence')
    report = dict(checked_at=time.time(), count=len(evidence), all_equal=all(
        e['exact_equal'] for e in evidence), comparisons=evidence,
        semantics='actual cropped observation token IDs, same docs/title/order/wrapper/500-token crop')
    atomic(receipt, report)
    if not report['all_equal']:
        raise RuntimeError('Retriever parity failed before HTTP admission')
    print('GROUNDING_RETRIEVER_PARITY=' + json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-repo', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpu-uuids', nargs='+', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--records', required=True)
    p.add_argument('--data-cache', required=True)
    p.add_argument('--port', type=int, default=19000)
    args = p.parse_args()
    if len(args.gpu_uuids) not in (1, 2) or len(set(args.gpu_uuids)) != len(args.gpu_uuids):
        p.error('Assign exactly one or two distinct auxiliary GPU UUIDs')
    out, repo = Path(args.output), Path(args.source_repo)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'admission.json').exists():
        raise RuntimeError('An existing retriever attempt cannot be overwritten')
    measured = {name: sha(repo / name) for name in SOURCE_SHA}
    if measured != SOURCE_SHA:
        raise ValueError('Frozen historical retriever sources changed')
    held, indices, rows = lease_aux(args.gpu_uuids, args.port)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, indices))
    os.environ['HF_HOME'] = str(out / 'hf-cache')
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['PYTHONPATH'] = str(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, '/data/search-r1/retriever-gpu-lib')
    run_id = 'grounding-' + uuid.uuid4().hex
    atomic(out / 'admission.json', dict(pid=os.getpid(), run_id=run_id, at=time.time(),
        gpu_uuids=args.gpu_uuids, physical_indices=indices, inventory=rows,
        source_repo=str(repo), source_sha256=measured,
        endpoint='http://127.0.0.1:' + str(args.port) + '/retrieve',
        original_service_restarted=False, own_independent_service=True,
        parity_required_before_listen=True))
    spec = importlib.util.spec_from_file_location('grounding_pinned_retriever',
        repo / 'reproduction/prefix20_online/retriever.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if len(args.gpu_uuids) == 1:
        original_main = inspect.getsource(module.main)
        adapted_main = single_gpu_entrypoint(original_main)
        atomic(out / 'single-gpu-launch-adaptation.json', dict(
            source_sha256=hashlib.sha256(original_main.encode()).hexdigest(),
            adapted_sha256=hashlib.sha256(adapted_main.encode()).hexdigest(),
            edits=['allow one assigned auxiliary GPU',
                   'accept single plain GpuIndex after visible-device count validation'],
            retrieval_computation_changed=False, parity_required_before_listen=True))
        exec(compile(adapted_main, '<grounding-single-gpu-main>', 'exec'), module.__dict__)
    original = module.BatchedRequests
    class ParityChecked(original):
        def __init__(self, retriever, **kwargs):
            verify_parity(retriever, records=args.records, tokenizer_path=args.tokenizer,
                          receipt=out / 'parity.json')
            super().__init__(retriever, **kwargs)
    module.BatchedRequests = ParityChecked
    sys.argv = [str(repo / 'reproduction/prefix20_online/retriever.py'),
        '--index', '/data/search-r1/assets/wiki/e5_Flat.index',
        '--corpus', '/data/search-r1/assets/wiki/wiki-18.jsonl',
        '--encoder', '/data/search-r1/assets/e5-base-v2',
        '--cache', str(out / 'cache'), '--data-cache', args.data_cache,
        '--run-id', run_id, '--port', str(args.port), '--batch-queries', '1024',
        '--gpu-shards', str(len(args.gpu_uuids))]
    # All held leases remain live until this own service exits.
    try:
        module.main()
    finally:
        for fd in held:
            os.close(fd)


if __name__ == '__main__':
    main()
