"""Inspect or once extend only accepted745501's Slurm TimeLimit; never submit."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
BODY='''import pathlib,json,hashlib,subprocess,time,re,os
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1')
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def run(args):
 try:
  r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);return dict(args=args,returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-1000:])
 except subprocess.TimeoutExpired:return dict(args=args,status='BOUNDED_TIMEOUT_RESULT_UNKNOWN')
def snapshot():
 o=dict(epoch=time.time(),job=run(['scontrol','show','job','-o','745501']),queue=run(['squeue','-h','-j','745501','-o','%i|%T|%M|%l|%L|%R']))
 for n in('progress','latest','failure','interrupted','complete'):
  p=b/(n+'.json')
  if p.exists():o[n]=read(p)
 return o
def validate(s):
 raw=s['job']['stdout'];assert s['job']['returncode']==0
 for item in('JobId=745501','JobName=q7b-native-joint124194','UserId=mou176','Account=mou176','QOS=normal','Partition=acd_u','NumCPUs=8','MinMemoryNode=128G','Command='+str(b/'training.sbatch'),'WorkDir='+str(b)):
  assert item in raw,item
 assert 'gres/gpu=1'in raw or'TresPerNode=gres/gpu:1'in raw
 assert read(b/'submission-result.json')['job_id']=='745501'
 assert proof(b/'config.json')['sha256']=='5aaa63c9348f410357be92ce491ba6d18286d618ab86dae6e63f193f2b4367b6'
 assert proof(b/'training.sbatch')['sha256']=='073299d9f26bf858e6d61fbc8c12d6b6d726079385ff010796bb3e89a84e1d51'
 assert not(b/'failure.json').exists()
 return raw
before=snapshot();raw=validate(before)
partition=run(['scontrol','show','partition','acd_u','-o']);qos=run(['sacctmgr','-n','-P','show','qos','where','Name=normal','format=Name,MaxWall'])
out=dict(action=v['action'],before=before,partition=partition,normal_QOS=qos,only_requested_change='TimeLimit=2-00:00:00',no_code_config_resources_priority_or_writer_change=True)
if v['action']=='extend':
 assert 'JobState=RUNNING'in raw,'Only existing active main writer permitted'
 intent=b/'timelimit-48h-intent.json';receipt=b/'timelimit-48h-result.json'
 if intent.exists():out['mutation']='NOT_REPEATED_EXISTING_INTENT_RECONCILE_CURRENT_JOB_ONLY';out['existing_intent']=read(intent)
 elif 'TimeLimit=2-00:00:00'in raw:out['mutation']='ALREADY_48H_NO_UPDATE'
 else:
  assert 'TimeLimit=1-00:00:00'in raw,'Unexpected existing walltime; refuse'
  assert partition.get('returncode')==0 and 'MaxTime=7-00:00:00'in partition['stdout'],'Actual partition limit must permit48h'
  assert qos.get('returncode')==0,'QoS MaxWall must be readable before update'
  qr=[x.split('|')for x in qos['stdout'].splitlines()if x.split('|')[0]=='normal'];assert len(qr)==1
  limit=qr[0][1];assert limit in('','UNLIMITED','2-00:00:00','7-00:00:00'),'QoS wall limit not established to permit48h: '+limit
  data=dict(epoch=time.time(),job_id='745501',before=before,partition=partition,normal_QOS=qos,requested_timelimit='2-00:00:00',only_job_metadata=True)
  with intent.open('x')as f:json.dump(data,f);f.flush();os.fsync(f.fileno())
  out['mutation']=run(['scontrol','update','JobId=745501','TimeLimit=2-00:00:00'])
 out['after']=snapshot();validate(out['after']);out['actual_48h']= 'TimeLimit=2-00:00:00'in out['after']['job']['stdout']
 out['source_config_and_entry_unchanged']=True
 if not receipt.exists():
  with receipt.open('x')as f:json.dump(out,f);f.flush();os.fsync(f.fileno())
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('inspect','extend'),required=True);a=p.parse_args();compile(BODY,'timelimit-only-remote','exec')
 spec=importlib.util.spec_from_file_location('main_timelimit_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();body='v='+repr(dict(action=a.action))+'\n'+BODY;packed=base64.b64encode(zlib.compress(body.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
 if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-2000:])
 out=json.loads(r.stdout.splitlines()[-1]);Path(__file__).with_name('actual-timelimit-'+a.action+'.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
