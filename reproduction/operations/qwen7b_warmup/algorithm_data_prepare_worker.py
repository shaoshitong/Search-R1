"""CPU-only durable preparation for G10 validation or extra genuine training G5.

Run through the existing remote runtime. No Slurm submission or generation.
"""
import argparse,fcntl,hashlib,json,os,time,traceback
from pathlib import Path

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
CODE_STAGE=B/'algorithm-data-preparation-code-20261021'
FINAL=B/'confirmation-materialization-deploy-20261021/materialized/materialization.json'
FINAL_SHA='ff68f243b1daa7d8a617a2c7a825e1126d0771eee2e92b96a85dc5f9b4f2bfe3'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for value in iter(lambda:f.read(1024*1024),b''):h.update(value)
 return h.hexdigest()

def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def run(request):
 r=json.loads(Path(request).read_text());base=Path(r['base']);kind=r['kind']
 assert kind in ('validation','extra128') and base.parent==B/'algorithm-improvement-20261021'
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 base.mkdir(parents=True,exist_ok=True);lock=(base/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (base/'result.json').exists() and not (base/'materialized').exists(),'Existing output must be inspected; never reprepare'
 pub=json.loads((CODE_STAGE/'source-publication.json').read_text());code=Path(pub['code_root'])
 assert pub['commit']==r['collector_commit'] and str(code)==os.environ['QWEN7B_CODE_ROOT']
 manifest=json.loads((code/'_confirmation_source_manifest.json').read_text())
 assert sha(code/'_confirmation_source_manifest.json')==pub['source_manifest_sha256']
 for name,item in manifest['files'].items():assert sha(code/name)==item['sha256'],name
 assert sha(Path(__file__))==r['worker_sha256'] and sha(FINAL)==FINAL_SHA
 controller=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],kind=kind,started=time.time(),worker_sha256=r['worker_sha256'],driver_commit=r['driver_commit'],collector_commit=pub['commit'],no_GPU=True,no_submission=True,responses_generated=0)
 atomic(base/'controller.json',controller)
 try:
  from reproduction.operations.qwen7b_warmup.validation_g10_materialize import prepare,write_final_exclusion
  if kind=='validation':
   result=prepare(B,B/'selection-2048-20261020',FINAL,base/'materialized',code,pub['commit'],20261021,FINAL_SHA)
  else:
   exclusion=write_final_exclusion(FINAL,base/'final-exclusion-plan.json',FINAL_SHA)
   from reproduction.operations.qwen7b_warmup.extra_g5_prepare import prepare as prepare_extra
   result=prepare_extra(B,B/'selection-2048-20261020',Path(exclusion['plan']['path']),base/'materialized',code,pub['commit'],128,20261021,5)
  record=dict(status='ALGORITHM_DATA_PREPARATION_COMPLETE_NOT_GENERATED',kind=kind,result=result,worker=controller,finished=time.time(),responses_generated=0,jobs_submitted=0)
 except BaseException as e:
  record=dict(status='ALGORITHM_DATA_PREPARATION_FAILED_NO_RESTART',kind=kind,error=repr(e),traceback=traceback.format_exc(),worker=controller,finished=time.time(),responses_generated=0,jobs_submitted=0)
  atomic(base/'failure.json',record);raise
 finally:
  if 'record' in locals():atomic(base/'result.json',record)
 print(json.dumps(dict(status=record['status'],kind=kind,result_status=result['status'],base=str(base),responses_generated=0,jobs_submitted=0)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();run(a.request)
