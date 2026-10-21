"""Bounded read-only progress observation; never launches or changes training.

A new candidate directory can precede progress.json. Its unknown step is an
ordinary observation state, not a training failure or a reason to restart.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-outer', type=int, required=True)
    parser.add_argument('--next-outer', type=int)
    parser.add_argument('--observations', type=int, default=40)
    parser.add_argument('--interval', type=int, default=45)
    args = parser.parse_args()
    if not (1 <= args.completed_outer <= 200 and 1 <= args.observations <= 80 and 10 <= args.interval <= 60):
        parser.error('Require outer1..200, observations1..80 and interval10..60')
    if args.next_outer is not None and not args.completed_outer < args.next_outer <= 200:
        parser.error('The next outer must be later than the target and at most200')
    repo = Path(__file__).resolve().parents[3]
    workspace = repo.parents[1]
    path = workspace/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py'
    spec = importlib.util.spec_from_file_location('existing_bound_read_only_bridge', path)
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    code = Path(__file__).with_name('observe_training_progress.py').read_text()
    command = shlex.join(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python', '-B', '-c', code])
    output = workspace/'outputs/easy2560-qwen7b-20261021'
    previous = None
    for index in range(args.observations):
        result = bridge.run('jxu957', command, '', str(workspace/'work'), login_host='10.120.48.31',
            expected_host_identity=dict(hostname='ACD-Login-1', boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',
                                        pid_namespace='pid:[4026531836]'), observation_timeout=55)
        receipt = output/('training-progress-em-'+str(int(time.time()))+'.json')
        receipt.write_text(json.dumps(dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), indent=2)+'\n')
        if result.returncode:
            print(json.dumps(dict(observation_unknown=True, receipt=receipt.name, returncode=result.returncode,
                                  training_failure_inferred=False)), flush=True)
        else:
            value = json.loads(result.stdout)
            last = (value['completed_outer_metrics'] or [{}])[-1]
            pool = (value['candidate_pools'] or [{}])[-1]
            checkpoint = value.get('preparation/rolling-checkpoints/latest.json', {}).get('step')
            pool_step = pool.get('step')
            errors = {key: re.findall(r'.*(?:Traceback|CUDA out of memory|RayTaskError|OutOfMemoryError|NCCLError|FloatingPointError).*',
                                      value[key]['tail']) for key in ('training_log', 'trainer_log')}
            statistics = value.get('service_health', {}).get('statistics', {})
            running = all(job+'|RUNNING|' in value['queue']['stdout'] for job in ('745371', '745198'))
            current = (checkpoint, last.get('step'), pool_step, pool.get('completed_questions', 0))
            if current != previous or any(errors.values()) or not running or statistics.get('failed_requests', 0):
                print(json.dumps(dict(epoch=value['epoch'], receipt=receipt.name, checkpoint=checkpoint,
                    completed_outer=last.get('step'), reward=last.get('reward'), pool_outer=pool_step,
                    questions=pool.get('completed_questions', 0), queries=statistics.get('completed_queries'),
                    failed=statistics.get('failed_requests'), errors=errors, jobs_running=running)), flush=True)
                previous = current
            complete = (type(checkpoint) is int and checkpoint >= args.completed_outer and
                        type(last.get('step')) is int and last['step'] >= args.completed_outer)
            next_saved = (args.next_outer is None or (type(pool_step) is int and pool_step >= args.next_outer and
                                                      pool.get('completed_questions', 0) > 0))
            if complete and next_saved:
                print('Requested actual checkpoint/update and saved-generation observations obtained; goal is not marked complete.', flush=True)
                return
            if any(errors.values()) or not running or statistics.get('failed_requests', 0):
                print('Inspection required; no training mutation performed.', flush=True)
                return
        if index+1 < args.observations:
            time.sleep(args.interval)
    print('Bounded observation window ended; production and goal unchanged.', flush=True)


if __name__ == '__main__':
    main()
