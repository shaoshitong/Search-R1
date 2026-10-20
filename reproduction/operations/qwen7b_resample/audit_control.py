"""Once-at-a-time CPU audit on NAS10; copies only this small verifier."""
import argparse,base64,hashlib,importlib.util,json,os,subprocess,time
from pathlib import Path
from .launch import ROOT
BASE='/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'

def main():
 p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--max-new-batches',type=int,default=25);a=p.parse_args();source=Path(__file__).with_name('audit_data.py').read_bytes()
 remote='BASE='+repr(BASE)+'\nSTART='+repr(a.start)+'\nLIMIT='+repr(a.max_new_batches)+'\nDATA='+repr(base64.b64encode(source).decode())+'\nSHA='+repr(hashlib.sha256(source).hexdigest())+'\n'+r'''
import base64,hashlib,json,pathlib,subprocess,time
b=pathlib.Path(BASE);root=b/'preparation/audit-control';root.mkdir(parents=True,exist_ok=True);active=root/'active.json';existing=json.loads(active.read_text())if active.exists()else None
status=''
if existing:status=subprocess.check_output(['systemctl','show',existing['unit'],'-p','MainPID','-p','Result','-p','ActiveState','-p','ExecMainStatus'],text=True)
v=dict(epoch=time.time(),existing=existing,unit_status=status)
if START and 'MainPID=0\n'not in status and existing:v['not_started']='existing audit is still active'
elif START:
 code=root/SHA;code.mkdir(exist_ok=True);f=code/'audit_data.py';data=base64.b64decode(DATA);assert hashlib.sha256(data).hexdigest()==SHA
 if not f.exists():f.write_bytes(data)
 assert hashlib.sha256(f.read_bytes()).hexdigest()==SHA
 (b/'audit').mkdir(exist_ok=True);tag=str(time.time_ns());unit='sst-q7b-data-audit-'+tag;log=root/(tag+'.log');intent=dict(unit=unit,epoch=time.time(),source_sha256=SHA,max_new_batches=LIMIT,log=str(log),gpu_used=False)
 with(root/(tag+'-intent.json')).open('x')as out:json.dump(intent,out)
 cmd=['systemd-run','--unit='+unit,'--setenv=CUDA_VISIBLE_DEVICES=','--property=CPUQuota=100%','--property=MemoryMax=2G','--property=SuccessExitStatus=2','--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log),'flock','-n',str(b/'audit/run.lock'),'/usr/bin/python3',str(f),'--base',str(b),'--max-new-batches',str(LIMIT)]
 r=subprocess.run(cmd,text=True,capture_output=True);v['submission']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,intent=intent);(root/(tag+'-receipt.json')).write_text(json.dumps(v['submission']));active.write_text(json.dumps(intent));assert r.returncode==0
q=b/'audit/latest.json'
if q.exists():
 x=json.loads(q.read_text());v['audit']={k:z for k,z in x.items()if k not in ['unsealed_inputs','sealed_pending_inputs','errors']};v['audit']['first_errors']=x.get('errors',[])[:5]
print(json.dumps(v))
'''
 for n,p in [('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py')]:
  s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);globals()[n]=m
 from .management_session import apply
 r=subprocess.run(apply(ctl.route('h100-10','/usr/bin/python3 -')),input=remote,text=True,capture_output=True,timeout=60);assert r.returncode==0,r.stderr;value=json.loads(r.stdout);stamp=time.time_ns();(ROOT/'work'/('qwen7b-audit-control-%d.json'%stamp)).write_text(json.dumps(value,indent=2));print(json.dumps(value))
if __name__=='__main__':main()
