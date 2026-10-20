"""Reclaim failed, unconsumed hidden tensors after complete CPU qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--outer', type=int, required=True)
    parser.add_argument('--failed-stage', choices=['score','expanded-continuation-rerank'], default='score')
    parser.add_argument('--live-base')
    args = parser.parse_args()
    source = Path(args.source)
    if source.parent != Path('/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong') or not source.name.startswith('search-g1-prefix-selection200-'):
        raise ValueError('Current retained18 experiment namespace required')
    def read(path):
        return json.loads(path.read_text())
    joint_path = source/'latest-joint.json'
    joint = read(joint_path)
    outer = args.outer
    proof = read(source/('preparation/recovery%d-cpu/full-state-proof.json'%(outer-1)))
    assert joint['outer'] == outer-1 == proof['outer']
    assert proof['status'] == 'DIVERSE_NATIVE_FULL_STATE_CPU_PASS'
    assert hashlib.sha256(joint_path.read_bytes()).hexdigest() == proof['joint_sha256']
    assert read(source/'logs/cleanup-status.json')['owned_births_all_gone']
    assert read(source/'actor/latest.json') == joint['actor']
    assert read(source/'probe/latest.json') == joint['probe']
    root = source/'outers'/('outer-%06d'%outer)
    assert not (root/'complete.json').exists()
    assert not list((source/'actor/outers').glob('outer-%04d-attempt-*'%outer))
    assert not (root/'probe-update').exists()
    assert read(source/'logs/pipeline-status.json')['stage'] == 'outer-%06d-%s'%(outer,args.failed_stage)
    if args.live_base:
        live=Path(args.live_base)
        assert live.parent==source.parent and live!=source
        live_joint=read(live/'latest-joint.json')
        assert live_joint['run_id']==joint['run_id'] and live_joint['outer']>=outer+7
        assert not(live/'logs/cleanup-status.json').exists()
        descendant=live; visited=set()
        while descendant!=source:
            assert descendant not in visited and descendant.parent==source.parent
            visited.add(descendant)
            config=read(descendant/'preparation/launch-config.json')
            assert config['run_id']==joint['run_id']
            descendant=Path(config['resume_joint']['source_base'])
        assert len(visited)>=1
    elif args.failed_stage!='score':
        raise ValueError('Later recovery lineage required for abandoned continuation features')
    scan = read(root/'prefix/scan/scan-complete.json')
    assert scan['run_id'] == joint['run_id'] and scan['actor_hidden_native']
    rows = []
    for item in scan['actor_hidden_shards']:
        f = Path(item['path'])
        assert f.parent == root/'prefix' and f.name.startswith('rank-') and f.name.endswith('-actor-hidden.npy')
        assert not f.is_symlink() and f.is_file()
        st = f.stat()
        assert st.st_size == item['bytes'] and len(item['sha256']) == 64
        rows.append(dict(path=str(f), bytes=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns,
            producer_sha256=item['sha256'], contents_rehashed=False))
    assert len(rows) == 8 and len({r['path'] for r in rows}) == 8
    intent = root/'abandoned-prefix-hidden-prune-intent.json'
    result = root/'abandoned-prefix-hidden-pruned.json'
    with intent.open('x') as handle:
        json.dump(dict(created_unix=time.time(),reason='Failed before optimizer work; pending sampling will be recomputed',
            qualified_joint=outer-1,actor_writes=0,probe_writes=0,assets=rows,
            failed_stage=args.failed_stage,live_base=args.live_base,
            keep='checkpoint, tokens, trajectories, diversity, scan manifests, failure logs and costs'),handle)
    for row in rows:
        f = Path(row['path']);st = f.stat()
        assert (st.st_ino,st.st_size,st.st_mtime_ns)==(row['inode'],row['bytes'],row['mtime_ns'])
        f.unlink()
    value = dict(completed_unix=time.time(),qualified_outer=outer-1,failed_outer=outer,
        removed_bytes=sum(r['bytes'] for r in rows),removed_count=len(rows),
        checkpoint_or_optimizer_deleted=False,assets=rows)
    with result.open('x') as handle:json.dump(value,handle)
    print(json.dumps(value),flush=True)


if __name__ == '__main__':
    main()
