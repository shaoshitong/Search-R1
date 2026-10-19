"""One two-GPU allocation: native 7B sampling + separate FP16 retriever GPU.

Each bounded batch finishes prefix/native features and SAME-G5 continuation,
then publishes its immutable complete manifest. An interrupted unsealed batch
is retained separately; accepted batches are never regenerated on restart.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from .download_model import atomic, fingerprints, REVISION


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return fingerprints(Path(path))['sha256']


def phase(config_path, mode):
    from . import sampling
    c = read(config_path)
    root = sampling.root_dir(c, 1)
    output = root / ('prefix' if mode == 'prefix' else 'full')
    output.mkdir(parents=True, exist_ok=True)
    receipt = output/'rank-0-complete.json'
    if not receipt.exists():
        abandoned = [p for p in output.iterdir() if p.name.startswith('rank-0-')]
        if abandoned:
            quarantine = output / ('unsealed-preserved-%d' % time.time_ns())
            quarantine.mkdir()
            for p in abandoned:
                shutil.move(str(p), str(quarantine/p.name))
    row = sampling.worker(config_path, 1, mode, 0, root/'probe-score/selection.json')
    for key in ('states','prefix','actor_hidden') if mode == 'prefix' else ('full','own_em'):
        item = row[key]
        assert sha(item['path']) == item['sha256']
    if mode == 'prefix':
        feature = read(row['actor_hidden']['path'])
        for item in feature['files'].values():
            assert sha(item['path']) == item['sha256']
        scan = dict(ranks=[row], head_prefix_tokens=c['capture_prefix_tokens'], model_revision=REVISION,
            questions=row['questions'], trajectories=row['trajectories'], feature_layer=26)
        scan_path = output/'scan.json'
        atomic(scan_path, scan)
        complete = dict(scan, scan_sha256=sha(scan_path), interaction_protocol=row['interaction_protocol'],
            actual_compute_dtype=row['actual_compute_dtype'])
        atomic(output/'complete.json', complete)
        groups = {}
        with gzip.open(row['prefix']['path'],'rt') as handle:
            for line in handle:
                prefix = json.loads(line)
                groups.setdefault(prefix['qid'], []).append(prefix['trajectory_id'])
        expected = read(root/'questions.json')['rows']
        assert set(groups) == {g['qid'] for g in expected} and all(len(v) == 5 for v in groups.values())
        selection = dict(run_id=c['run_id'],outer=1,actor_version=c['actor_version'],
            prefix_scan_sha256=complete['scan_sha256'],kind='ALL_G5_NO_SELECTION_NO_OVERSAMPLING',
            selected=[dict(qid=g['qid'],question_hash=g['question_hash'],candidate_ordinal=g['candidate_ordinal'],trajectory_ids=groups[g['qid']]) for g in expected])
        (root/'probe-score').mkdir(exist_ok=True)
        atomic(root/'probe-score/selection.json', selection)
    else:
        labels = {}
        with Path(row['own_em']['path']).open() as handle:
            for line in handle:
                item = json.loads(line)
                assert item['trajectory_id'] not in labels and item['strict_em'] in (0,1)
                labels[item['trajectory_id']] = item['strict_em']
        selection = read(root/'probe-score/selection.json')
        assert set(labels) == {tid for g in selection['selected'] for tid in g['trajectory_ids']}
        atomic(output/'complete.json', dict(row, labels=len(labels), warmup_only=True,
            actor_optimizer_updates=0, model_revision=REVISION))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--shard', required=True, type=int)
    args = parser.parse_args()
    c = read(args.config)
    assert c['warmup_only'] and c['model_revision'] == REVISION
    base = Path(c['base'])
    root = base/'sampling'/('shard-%02d' % args.shard)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assigned = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(assigned) == 2 and len(set(assigned)) == 2
    proof = read(Path(c['model_dir'])/'complete.json')
    assert proof['status'] == 'MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' and proof['revision'] == REVISION
    atomic(root/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        job_id=os.environ['SLURM_JOB_ID'], epoch=time.time(), assigned_devices=assigned, generation_devices=1,retrieval_devices=1))
    batches = sorted((base/'inputs'/('shard-%02d' % args.shard)).glob('batch-*.json'))
    assert batches
    completed = [p for p in (root/'batches').glob('*/complete.json')] if (root/'batches').exists() else []
    retriever = None
    log = (root/'retriever.log').open('ab')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=assigned[1], OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    port = 25000 + int(os.environ['SLURM_JOB_ID']) % 20000
    retriever_cache = root/'retriever'
    command = [sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.retriever',
        '--index',c['retrieval_index'],'--corpus',c['retrieval_corpus'],'--encoder',c['retrieval_encoder'],
        '--cache',str(retriever_cache),'--data-cache',c['retrieval_data_cache'],'--run-id',c['run_id'],
        '--port',str(port),'--gpu-shards','1','--batch-queries','256']
    retriever = subprocess.Popen(command,env=env,stdout=log,stderr=log)
    started = time.monotonic()
    try:
        while True:
            if retriever.poll() is not None:
                raise RuntimeError('Owned retriever exited during initialization')
            try:
                with urllib.request.urlopen('http://127.0.0.1:%d/health' % port,timeout=2) as response:
                    health=json.load(response)
                if health.get('ok') or health.get('ready'):
                    break
            except Exception:
                pass
            if time.monotonic()-started > 1800:
                raise TimeoutError('Retriever initialization exceeded bound')
            time.sleep(2)
        for batch_path in batches:
            batch = read(batch_path)
            dest = root/'batches'/batch_path.stem
            if (dest/'complete.json').exists():
                accepted = read(dest/'complete.json')
                assert accepted['input_sha256'] == sha(batch_path) and accepted['run_id'] == c['run_id']
                continue
            outer = dest/'outers/outer-000001'
            outer.mkdir(parents=True,exist_ok=True)
            atomic(outer/'questions.json',dict(outer=1,rows=batch['rows']))
            config = dict(c, base=str(dest), search_url='http://127.0.0.1:%d/retrieve' % port,
                actor_model_dir=c['model_dir'],actor_version='official7b-'+REVISION,
                actor_hidden_native=True,world_size=1,prefix_policy_tokens=c['capture_prefix_tokens'],
                rollout_max_num_seqs=256,rollout_max_batched_tokens=32768,
                rollout_memory_utilization=.90,rollout_enforce_eager=True,rollout_retry_initialization_oom=False)
            path = dest/'config.json'
            atomic(path,config)
            prefix = phase(path,'prefix')
            full = phase(path,'continue')
            accepted = dict(run_id=c['run_id'], input_sha256=sha(batch_path), batch=batch_path.stem,
                questions=len(batch['rows']),trajectories=5*len(batch['rows']),prefix=prefix,full=full,
                feature_layer=26, model_revision=REVISION, actor_optimizer_updates=0, epoch=time.time())
            atomic(dest/'complete.json',accepted)
            completed.append(dest/'complete.json')
            atomic(root/'progress.json',dict(completed_batches=len(completed),total_batches=len(batches),
                questions=sum(read(p)['questions'] for p in completed),trajectories=sum(read(p)['trajectories'] for p in completed),
                epoch=time.time(), elapsed=time.monotonic()-started))
            print(json.dumps(dict(stage='REAL_G5_BATCH_COMPLETE',batch=batch_path.stem,questions=accepted['questions'],trajectories=accepted['trajectories'])),flush=True)
        atomic(root/'complete.json',dict(status='ALL_ASSIGNED_CANONICAL_G5_COMPLETE',shard=args.shard,
            run_id=c['run_id'],batches=[str(p) for p in completed],questions=sum(read(p)['questions'] for p in completed),
            trajectories=sum(read(p)['trajectories'] for p in completed),model_revision=REVISION,epoch=time.time()))
    except BaseException as error:
        atomic(root/'failure.json',dict(error_type=type(error).__name__,error=str(error)[:300],epoch=time.time()))
        raise
    finally:
        # Only this allocation's explicitly created child is signalled.
        if retriever is not None and retriever.poll() is None:
            retriever.terminate()
            try:retriever.wait(timeout=20)
            except subprocess.TimeoutExpired:retriever.kill();retriever.wait()
        log.close()


if __name__ == '__main__':
    main()
