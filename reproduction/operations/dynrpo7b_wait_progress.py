"""Bounded read-only observation of the explicitly bound training job.

A timeout or SSH failure is an observation result, never a training failure and
never a reason to restart. Only small local summaries are persisted.
"""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from reproduction.operations.dynrpo7b_watch_recoveryfix_r6 import remote, observation_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until-outer', type=int, required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base')
    parser.add_argument('--training-job', type=int, default=751884)
    parser.add_argument('--retrieval-job', type=int, default=751011)
    args = parser.parse_args()
    script = observation_script(args.base, args.training_job, args.retrieval_job)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                remote(script)
            state = json.loads(captured.getvalue())
        except (Exception, SystemExit) as error:
            print(json.dumps({'status': 'OBSERVATION_FAILED_NOT_TERMINAL',
                              'error_type': type(error).__name__}), flush=True)
            time.sleep(50)
            continue
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(args.output)
        complete = state.get('checkpoint', {}).get('completed_outer', 0)
        brief = {key: state.get(key) for key in ('time', 'queue', 'writes', 'latest_collection_journal')}
        brief['completed_outer'] = complete
        brief['training_error'] = any(state.get(key, {}).get('error', False)
                                      for key in ('trainer', 'launcher'))
        print(json.dumps(brief), flush=True)
        if brief['training_error'] or state.get('terminal'):
            print(json.dumps({'status': 'INSPECT_ACTUAL_TRAINING_STATE',
                              'terminal': state.get('terminal')}), flush=True)
            return
        if complete >= args.until_outer:
            print(json.dumps({'status': 'OBSERVED_REQUESTED_OUTER', 'completed_outer': complete}), flush=True)
            return
        time.sleep(50)
    print(json.dumps({'status': 'OBSERVATION_TIMEOUT_NOT_TERMINAL'}), flush=True)


if __name__ == '__main__':
    main()
