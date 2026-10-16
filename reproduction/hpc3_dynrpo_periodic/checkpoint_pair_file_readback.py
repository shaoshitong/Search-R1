"""Remote CPU-only fresh file SHA readback of an existing accepted checkpoint pair."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', required=True, type=Path)
    parser.add_argument('--pair-sha256', required=True)
    parser.add_argument('--outer', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicitly disable CUDA for file SHA inspection')
    raw = args.pair.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.pair_sha256
    pair = json.loads(raw)
    assert pair['step'] == args.outer and pair['world_size'] == 4
    assert sorted(r['rank'] for r in pair['ranks']) == list(range(4))
    assert all(r['completed_outer'] == args.outer and r['actual_optimizer_writes'] == 20*args.outer for r in pair['ranks'])
    base = Path(pair['source_base']).resolve()
    assert args.output.resolve().is_relative_to(base/'logs/manual-verification')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Reuse the immutable experiment's existing bounded streaming hash code.
        from search_g1.qprobe_checkpoint import _file_record
        records = list(pair['files']) + list(pair['boundary_worker_files'])
        records += [pair['boundary_driver'], pair['source_run_manifest'], pair['reward_probe_manifest']]
        records += list(pair['reward_files']) + list(pair.get('question_probe_files', []))
        unique = {}
        for record in records:
            path = record['path']
            if path in unique:
                assert unique[path]['full_read_sha256'] == record['full_read_sha256']
            unique[path] = record
        verified = []
        started = time.time()
        for path, expected in unique.items():
            current = _file_record(path)
            assert all(current[k] == expected[k] for k in ('bytes','inode','mtime_ns','full_read_sha256'))
            verified.append(dict(path=path, bytes=current['bytes'], sha256=current['full_read_sha256']))
        assert args.pair.read_bytes() == raw
        result = dict(status='FRESH_CHECKPOINT_AND_PAIR_FILES_SHA_VERIFIED', checked_unix=time.time(),
            elapsed_seconds=time.time()-started, outer=args.outer, pair=str(args.pair),
            pair_sha256=args.pair_sha256, execution_phase=pair['execution_phase'],
            checkpoint_file_count=len(pair['files']), unique_pair_file_count=len(verified),
            full_read_bytes=sum(r['bytes'] for r in verified), files=verified, ranks=pair['ranks'],
            saved_state_cpu_check_scope=pair['scope'], actual_cuda_restore_accepted=False,
            scope='Fresh streaming SHA/stat of checkpoint, driver/RNG and probe/reward bindings; CPU saved-state validation is from the accepted pair; no GPU reload or optimizer update')
        temporary = args.output.with_name(args.output.name+'.tmp-'+uuid.uuid4().hex)
        temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(args.output)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
