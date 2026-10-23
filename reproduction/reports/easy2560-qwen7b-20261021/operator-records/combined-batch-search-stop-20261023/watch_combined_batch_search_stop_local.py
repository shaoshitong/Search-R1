"""Read one search-stop continuation snapshot through the existing transport."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    payload = Path(__file__).with_name('observe_combined_batch_search_stop_hpc.py')
    call = subprocess.run(
        [sys.executable, '-B', '/tmp/easy7b_diag_remote.py', str(payload),
         'combined-batch-search-stop-transition-live'], capture_output=True, text=True)
    paths = [line for line in call.stdout.splitlines()
             if line.startswith('/Users/') and line.endswith('.json')]
    if not paths:
        print(call.stdout[-2000:])
        print(call.stderr[-1000:])
        return call.returncode or 1
    receipt = Path(paths[0])
    outer = json.loads(receipt.read_text())
    if outer['returncode']:
        print(json.dumps(dict(evidence=str(receipt), returncode=outer['returncode'],
                              error=outer['stderr'][-2000:])))
        return outer['returncode']
    snapshot = json.loads(outer['stdout'])
    keys = ('epoch', 'base', 'job', 'queue', 'accounting',
            'requested_runtime_commit', 'actual_runtime_commit', 'runtime_commit_matches',
            'source_job', 'source_completed_outer', 'restoration_observed',
            'latest_completed_outer', 'post_restore_completed_outer_count',
            'optimizer_ranks', 'candidate_progress', 'selection_summary',
            'transition_documents', 'actual_error_lines', 'observation_errors')
    result = {key: snapshot.get(key) for key in keys}
    result['evidence'] = str(receipt)
    latest = snapshot.get('documents', {}).get('preparation/rolling-checkpoints/latest.json', {})
    result['rolling_checkpoint'] = latest.get('value')
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
