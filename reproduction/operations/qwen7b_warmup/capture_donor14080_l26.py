"""Future one-GPU sequential four-shard capture; this file submits nothing."""
import argparse,fcntl,json,os
from prepare_donor14080_l26 import TARGET,QUALIFIED,PUB,DONOR_COMPLETE,TAIL_COMPLETE,fp,checked,read,qualified
STATUS='DONOR14080_SOURCE_RESOLVED_REAL_G10_QUALIFIED_HF_LAYER26_SHARD_COMPLETE'
def capture(preparation_sha):
 import torch
 if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1:raise ValueError('One owned GPU allocation is required; this entry never requests one')
 pub,r,core=qualified();pr=fp(TARGET/'preparation.json');assert pr['sha256']==preparation_sha;prep=read(pr['path']);assert prep['questions']==14080 and prep['trajectories']==140800 and prep['qualified_runtime_publication']==PUB
 checked(DONOR_COMPLETE);checked(TAIL_COMPLETE);checked(prep['source_resolution_contract']);checked(prep['physical_source_final_readback'])
 with(TARGET/'single-gpu-capture.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);results=[]
  for i,ref in enumerate(prep['selections']):
   sel=read(checked(ref));assert sel['shard']==i and sel['questions']==3520 and sel['source_ingest']==DONOR_COMPLETE and sel['expected_code_root']==str(QUALIFIED)
   checked(sel['chosen_physical_prefix_bindings']);core.capture(ref['path'],ref['sha256'],TARGET/('shard-%02d'%i)/'capture',expected_questions=3520,complete_status=STATUS);results.append(fp(TARGET/('shard-%02d'%i)/'capture/complete.json'))
  return dict(status='DONOR14080_FOUR_LAYER26_SHARDS_CAPTURED_PENDING_RETAINED_ROW_SOURCE_BANK_ACCEPTANCE',preparation=pr,shards=results,actor_updates=0,probe_updates=0,new_responses_generated=0,job_id=os.environ['SLURM_JOB_ID'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preparation-sha256',required=True);a=p.parse_args();print(json.dumps(capture(a.preparation_sha256),sort_keys=True))
