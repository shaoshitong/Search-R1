"""CPU-only exact-cohort authentic extra-G5 preparation; never submits Slurm.

The collector is a separately published immutable import closure. A durable
request binds its commit, the predeclared matched question plan, and the frozen
final exclusions. Existing output/intent is never silently overwritten.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

PARENT=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
FINAL_SHA='ff68f243b1daa7d8a617a2c7a825e1126d0771eee2e92b96a85dc5f9b4f2bfe3'
PLAN_SHA='3423f65455a60c76d38d09802502e6bc343205fb3ffa2316541e35b80f7d195a'
COLLECTOR_COMMIT='2b293d6e6a0be5a8b99f976b3ac0c2b15e19ee40'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def atomic(p,value):
    p=Path(p);tmp=p.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def run(request_path):
    request_path=Path(request_path).resolve();r=json.loads(request_path.read_text())
    base=Path(r['base']).resolve()
    if base.parent!=PARENT/'algorithm-improvement-20261021' or request_path!=base/'request.json':
        raise ValueError('Isolated authorized remote preparation root required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation is CPU-only')
    if sha(__file__)!=r['worker_sha256']:raise ValueError('Published worker SHA differs')
    pub=json.loads((PARENT/'matched-g10-collector-code-20261021/source-publication.json').read_text())
    code=Path(pub['code_root'])
    if pub['commit']!=COLLECTOR_COMMIT or str(code)!=os.environ['QWEN7B_CODE_ROOT']:
        raise ValueError('Immutable collector identity differs')
    final=PARENT/'confirmation-materialization-deploy-20261021/materialized/materialization.json'
    plan=PARENT/'algorithm-improvement-20261021/matched-loss128-v2/plan.json'
    if sha(final)!=FINAL_SHA or sha(plan)!=PLAN_SHA:raise ValueError('Frozen plan SHA differs')
    with (base/'prepare.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'result.json').exists() or (base/'materialized').exists():
            raise ValueError('Existing preparation output must be reconciled, never restarted')
        controller=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            started=time.time(),worker_sha256=r['worker_sha256'],collector_commit=pub['commit'],
            driver_commit=r['driver_commit'],actor_updates=0,GPU_used=False,jobs_submitted=False)
        atomic(base/'controller.json',controller)
        try:
            from reproduction.operations.qwen7b_warmup.validation_g10_materialize import write_final_exclusion
            from reproduction.operations.qwen7b_warmup.extra_g5_prepare import prepare
            exclusions=write_final_exclusion(final,base/'final-exclusion-plan.json',FINAL_SHA)
            result=prepare(PARENT,PARENT/'selection-2048-20261020',exclusions['plan']['path'],
                base/'materialized',code,pub['commit'],6528,20261021,5,plan,PLAN_SHA)
            result=dict(status='MATCHED6528_EXACT_COHORT_EXTRA_REAL_G5_PREPARED_NOT_GENERATED',
                result=result,controller=controller,completed=time.time(),actor_updates=0,
                GPU_used=False,jobs_submitted=False,matched_plan_sha256=PLAN_SHA)
            atomic(base/'result.json',result);print(json.dumps(result),flush=True)
        except BaseException as e:
            atomic(base/'failure.json',dict(status='CPU_PREPARATION_FAILED_DO_NOT_BLINDLY_RESTART',
                error=repr(e),traceback=traceback.format_exc(),time=time.time(),controller=controller))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();run(a.request)
