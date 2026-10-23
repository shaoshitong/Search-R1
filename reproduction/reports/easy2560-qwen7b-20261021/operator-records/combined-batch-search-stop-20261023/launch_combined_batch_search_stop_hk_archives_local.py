"""Launch the existing HK core/companion pair for an explicitly supplied new job.

Prepared locally only. --print-plan reads no credentials and makes no connection.
Actual invocation reuses the published b8df9cc launcher and RAM-only password input.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys
import time

WORKSPACE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023'
RUNTIME = 'dde1452d2ec65fb3f4a0464d4beacf3b8136414c'
CONTROLLER = 'b8df9cc12904ae6729768107ff4aa529cff666af'
HK = '/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017'
RUN_ID = '232f20e112ac4a978c5aba52a0367195'
LOGIN_IDENTITY = dict(hostname='ACD-Login-1', boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',
                      pid_namespace='pid:[4026531836]')


def plan(job):
    return dict(base=BASE, training_commit=RUNTIME, source_job=job, run_id=RUN_ID,
                controller_commit=CONTROLLER, first_step=100, requested_steps=[100, 150, 200],
                launch_entry=HK+'/source/'+CONTROLLER+'/reproduction/operations/easy7b_20261021/fa3_archive_consumers_hk.py',
                login_host='10.120.48.31', login_identity=LOGIN_IDENTITY,
                core=HK+'/archive-controller/job-'+job,
                companion=HK+'/post-callback-boundary-control/job-'+job+'/future-100-200',
                old_consumers_modified=False, training_modified=False,
                credentials_persisted=False)


def remote_code():
    return '''import sys,json,importlib.util
from pathlib import Path
payload=json.load(sys.stdin)
p=Path(payload['launch_entry'])
spec=importlib.util.spec_from_file_location('search_stop_consumers',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.launch(payload['base'],payload['training_commit'],payload['source_job'],payload.pop('password'),payload['controller_commit'],first_step=payload['first_step'],login_host=payload['login_host'],login_identity_json=json.dumps(payload['login_identity']))
print(json.dumps({'status':r['status'],'job':r['source_job'],'requested_steps':r['requested_steps'],'core':r['core']['birth'],'companion':r['companion']['birth']}),flush=True)
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, help='New search-stop Slurm training job ID; unknown until submitted')
    parser.add_argument('--print-plan', action='store_true', help='Local parameters only; no credentials or remote calls')
    args = parser.parse_args(argv)
    if not args.job.isdecimal() or args.job == '750029':
        parser.error('--job must be the new numeric search-stop job ID, never source job 750029')
    values = plan(args.job)
    if args.print_plan:
        print(json.dumps(values, indent=2))
        return 0
    spec = importlib.util.spec_from_file_location('campus_ctl', WORKSPACE/'work/start_cosearch_flash_hf.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    secret = Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
    line = next(line for line in secret.splitlines() if line.startswith('账号 jxu957'))
    password = re.split(r'密码\s*[:：]?\s*', line, 1)[1].strip().strip('`')
    secret = line = None
    values['password'] = password
    try:
        result = transport.campus('hkust3', shlex.join(['python3', '-B', '-c', remote_code()]),
                                  input=json.dumps(values), observation_timeout=55)
    finally:
        values.pop('password', None)
        password = None
    receipt = Path(__file__).with_name('combined-batch-search-stop-hk-archive-launch-'+str(time.time_ns())+'.json')
    receipt.write_text(json.dumps(dict(returncode=result.returncode, stdout=result.stdout,
                                       stderr=result.stderr), indent=2)+'\n')
    print(receipt)
    print(result.stdout)
    print(result.stderr[-2000:])
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
