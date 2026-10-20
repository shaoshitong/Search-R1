"""One bounded read of an already submitted native sampler; no state mutation."""
import argparse
import json
from pathlib import Path
import time
from . import follow_shared as follow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-pid', required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    assert args.controller_pid.isdigit() and args.label.replace('-', '').isalnum()
    script = '\n'.join(key + '=' + repr(value) for key, value in dict(
        PHYSICAL=follow.PHYSICAL, RUN_ID=follow.RUN_ID, PID=args.controller_pid,
        JOB_PREFIX='job-relay10-').items()) + '\n' + follow.REMOTE
    result = follow.transport(script)
    assert result.returncode == 0, result.stderr[-2000:]
    value = json.loads(result.stdout)
    output = follow.ROOT / 'work' / ('qwen7b-native-current-%s-actual-%d.json' % (args.label, time.time_ns()))
    output.write_text(json.dumps(value, indent=2) + '\n')
    small = dict(epoch=value['epoch'], controller_pid=args.controller_pid,
        sealed_batches=value['sealed_batches'], questions=value['questions'], trajectories=value['trajectories'],
        progress=value.get('progress'), phases=value.get('active_phases'),
        model_initialized=value.get('model_initialized'), failure=value.get('failure'),
        native_complete=value.get('complete'), actual=str(output), goal_complete=False)
    if value.get('failure'):
        small['log_tails'] = value.get('log_tails')
    print(json.dumps(small))


if __name__ == '__main__':
    main()
