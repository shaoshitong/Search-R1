"""One native4 diagnostic submission; never stops or restarts production."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    request = json.load(sys.stdin)
    commit = request['commit']
    if os.getuid() != 14090 or re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('Require current jxu957 and a published source commit')
    base = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-model-probe-20261021')
    source = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
    if any((base / name).exists() for name in ('job.id', 'preparation/model-probe-submit-claim.json',
                                               'preparation/model-probe-submission.json')):
        print(json.dumps(dict(status='EXISTING_MODEL_PROBE_INSPECT_NO_DUPLICATE', base=str(base))))
        return
    sys.path.insert(0, str(base / 'source' / commit))
    from reproduction.easy2560_joint4.control import bootstrap, publish
    from reproduction.easy2560_joint4.flash3_runtime import prepare as prepare_runtime
    from reproduction.easy2560_joint4.retrieval_proxy import external_binding
    bootstrap(base, commit, predecessor=source)
    runtime = prepare_runtime(base)
    external = base / 'preparation/external-retriever-receipt.json'
    if not external.exists():
        publish(external, json.loads((source / 'preparation/external-retriever-receipt.json').read_text()))
    service = external_binding(external)
    command = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
        '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:4',
        '--cpus-per-task=48', '--mem=512G', '--time=00:40:00',
        '--job-name=easy7b-fa3-model-probe', '--export=NONE', '--chdir=' + str(base / 'repo'),
        '--output=' + str(base / 'logs/model-probe-%j.log'),
        '--error=' + str(base / 'logs/model-probe-%j.log'),
        str(base / 'source' / commit / 'reproduction/operations/easy7b_20261021/run_fa3_model_probe.sh'),
        str(base), commit, str(source)]
    claim = dict(source_commit=commit, command=command, source_base=str(source),
        source_job=(source / 'job.id').read_text().strip(), external_retriever=service,
        private_fa3_runtime=runtime, diagnostic_only=True, production_modified=False,
        production_dispatch_allowed=False, created_unix=time.time())
    publish(base / 'preparation/model-probe-submit-claim.json', claim)
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    receipt = dict(claim, returncode=result.returncode, stdout=result.stdout,
                   stderr=result.stderr, checked_unix=time.time())
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            raise ValueError('Unknown submission result; inspect original claim')
        receipt['job'] = job
        with (base / 'job.id').open('x') as stream:
            stream.write(job + '\n'); stream.flush(); os.fsync(stream.fileno())
    publish(base / 'preparation/model-probe-submission.json', receipt)
    print(json.dumps(receipt), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
