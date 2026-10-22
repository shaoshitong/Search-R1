"""Terminal-only CPU wrapper around the actual published P80 checkpoint verifier."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from . import accept_online_mixed_epoch_boundary as core
BOUNDARY=core.OUT
OUT=core.A/'online-mixed-P80-terminal-CPU-acceptance-20261023-v1'

def terminal_gate(layer):
 base=core.SUITE/('layer%d-prefix80'%layer)
 if not(base/'complete.json').is_file():return None
 receipt=core.read(base/'root-submission.json');job=str(receipt['job_id']);controller=core.read(base/'controller.json');complete=core.read(base/'complete.json')
 if (str(controller['job_id'])!=job or controller['layer']!=layer or controller['prefix']!=80):raise ValueError('Current actual training controller/root job identity differs')
 if any(complete.get(k)!=v for k,v in dict(status='MATCHED_ONLINE_MIXED_LAYER_PREFIX_PILOT_COMPLETE',epochs=3,cursor=192,layer=layer,prefix=80).items()):return None
 r=subprocess.run(['sacct','-n','-X','-j',job,'--format=JobIDRaw,State%32,ExitCode','-P'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if r.returncode:raise RuntimeError('Read-only Slurm accounting unavailable: '+r.stderr[-600:])
 own=[x.split('|')for x in r.stdout.splitlines()if x.split('|')[0].strip()==job]
 if len(own)!=1 or own[0][1].strip()!='COMPLETED'or own[0][2].strip()!='0:0':return None
 return dict(job_id=job,Slurm_accounting=own[0][:3],root_receipt=core.sha(base/'root-submission.json'),controller=core.sha(base/'controller.json'),complete=core.sha(base/'complete.json'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--layer',type=int,choices=(15,18),required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
 if os.uname().sysname!='Linux'or os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise RuntimeError('Remote CPU-only acceptance required')
 if core.sha(Path(__file__))['sha256']!=a.source_sha:raise ValueError('Terminal wrapper source SHA differs')
 gate=terminal_gate(a.layer)
 if gate is None:print(json.dumps(dict(status='WAIT_ACTUAL_COMPLETE_E3_AND_CURRENT_JOB_COMPLETED_ZERO_EXIT',layer=a.layer,checkpoint_opened=False,Slurm_mutated=False,time=time.time())));return
 assert OUT.parent.is_dir()and BOUNDARY.is_dir()
 OUT.mkdir(exist_ok=True);lock=(BOUNDARY/'acceptance.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 core.OUT=OUT
 core.atomic(OUT/'controller.json',dict(pid=os.getpid(),layer=a.layer,prefix=80,expected_epoch=3,expected_cursor=192,job_id=gate['job_id'],source_sha=a.source_sha,global_CPU_lock=str(BOUNDARY/'acceptance.lock'),time=time.time()))
 try:
  result=core.validate_one(a.layer,80,3)
  after=terminal_gate(a.layer)
  if after!=gate:raise ValueError('Current terminal/source identity changed during CPU acceptance')
  result=dict(status='TERMINAL_P80_E3_CPU_SHA_RAW_EMA_ADAM192_TWO_RANK_RNG_ACCEPTED',terminal_gate=gate,published_checkpoint_CPU_proof=result,expected_epoch=3,expected_cursor=192,layer=a.layer,prefix=80,shared_global_CPU_lock=str(BOUNDARY/'acceptance.lock'),GPU_forward_or_next_update_tested=False,Slurm_mutated=False,source=core.sha(Path(__file__)),time=time.time())
  core.atomic(OUT/('layer%d-prefix80'%a.layer)/'terminal-accepted.json',result);print(json.dumps(result))
 except BaseException as e:core.atomic(OUT/'failure.json',dict(layer=a.layer,error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
