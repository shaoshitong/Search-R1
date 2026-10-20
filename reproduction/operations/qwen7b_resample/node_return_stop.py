"""User-authorized 01:50 stop of only r6; keep complete batches on NAS10."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
BASE=Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
UNIT='sst-q7b-newseed-g5-20261020-r6.service'

def birth(pid):
 try:return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except (OSError,IndexError):return None

def seal():
 c=json.loads((BASE/'config.json').read_text());assert c['run_id']=='70188b8e50fd4b45bc420ce58014f46e'
 out=BASE/'preparation/node-return-20261021-0150';out.mkdir(parents=True,exist_ok=True)
 if(out/'complete.json').exists():print((out/'complete.json').read_text());return
 pid=int(subprocess.check_output(['systemctl','show',UNIT,'-p','MainPID','--value'],text=True));owned={pid}if pid else set()
 links={}
 for p in Path('/proc').glob('[0-9]*/stat'):
  try:links[int(p.parent.name)]=int(p.read_text().rsplit(')',1)[1].split()[1])
  except (OSError,ValueError,IndexError):pass
 while True:
  extra={p for p,parent in links.items()if parent in owned};new=owned|extra
  if new==owned:break
  owned=new
 births={str(p):birth(p)for p in owned};before=dict(user_instruction='Keep 02:00 node return; stop this sampling at 01:50',unit=UNIT,run_id=c['run_id'],epoch=time.time(),owned_births=births,full_data_remain_on_nas10=True)
 (out/'intent.json').write_text(json.dumps(before))
 r=subprocess.run(['systemctl','stop',UNIT],text=True,capture_output=True,timeout=100)
 remaining={p:b for p,b in births.items()if b is not None and birth(int(p))==b};logs=[]
 if not remaining:
  for p in sorted((BASE/'logs/r6').glob('*.log')):
   st=p.stat();h=hashlib.sha256()
   with p.open('rb')as f:
    for block in iter(lambda:f.read(1024**2),b''):h.update(block)
   logs.append(dict(path=str(p),inode=st.st_ino,bytes=st.st_size,sha256=h.hexdigest()))
 rows=[json.loads(p.read_text())for p in (BASE/'sampling').glob('shard-*/batches/batch-*/complete.json')]
 result=dict(status='USER_NODE_RETURN_SAMPLING_STOP_SEALED'if not remaining and r.returncode==0 else 'STOP_NEEDS_ATTENTION',unit=UNIT,epoch=time.time(),questions=sum(x['questions']for x in rows),trajectories=sum(x['trajectories']for x in rows),complete_batches=len(rows),remaining_original_births=remaining,terminal_logs=logs,stop_returncode=r.returncode,service=subprocess.check_output(['systemctl','show',UNIT,'-p','MainPID','-p','Result','-p','ActiveState'],text=True),goal_complete=False,shared_data_deleted=False)
 (out/'complete.json').write_text(json.dumps(result));print(json.dumps(result))

def stage():
 import base64,importlib.util,shlex
 from .launch import ROOT
 source=Path(__file__).read_bytes();source_sha=hashlib.sha256(source).hexdigest()
 remote='SOURCE='+repr(base64.b64encode(source).decode())+'\nSHA='+repr(source_sha)+'\n'+r'''
import base64,hashlib,json,subprocess,time
from pathlib import Path
root=Path('/root/.local/share/search-g1/qwen7b-newseed-g5/r6');assert(root/'intent.json').is_file()
p=root/'node-return-stop-intent.json';timer='sst-q7b-before-node-return-r6'
if p.exists():print(json.dumps({'existing':json.loads(p.read_text()),'not_replayed':True}));raise SystemExit(0)
data=base64.b64decode(SOURCE);assert hashlib.sha256(data).hexdigest()==SHA;(root/'node_return_stop.py').write_bytes(data)
script=root/'stop-at-0150.sh';script.write_text(chr(10).join(['#!/bin/bash','set -euo pipefail','pid=$(systemctl show sst-q7b-newseed-g5-20261020-r6.service -p MainPID --value)','if [[ "$pid" = 0 ]]; then echo "Service already stopped; accepted batches remain on NAS10"; exit 0; fi','exec nsenter --mount --target "$pid" /usr/bin/python3 '+str(root/'node_return_stop.py')+' --seal','']))
v=dict(epoch=time.time(),deadline='2026-10-21 01:50:00 Asia/Shanghai',only_unit='sst-q7b-newseed-g5-20261020-r6.service',source_sha256=SHA,keep_existing_0200_wipe=True)
with p.open('x')as f:json.dump(v,f)
r=subprocess.run(['systemd-run','--unit='+timer,'--on-calendar='+v['deadline'],'--timer-property=AccuracySec=1s','/bin/bash',str(script)],text=True,capture_output=True);result=dict(intent=v,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,timer=subprocess.check_output(['systemctl','show',timer+'.timer','-p','NextElapseUSecRealtime','-p','ActiveState'],text=True));(root/'node-return-stop-receipt.json').write_text(json.dumps(result));print(json.dumps(result));assert r.returncode==0
'''
 for n,p in [('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py'),('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py')]:
  spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);globals()[n]=m
 os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1';r=subprocess.run(route.management_argv(ctl.route('h100-11','/usr/bin/python3 -')),input=remote,text=True,capture_output=True,timeout=60);(ROOT/'work/qwen7b-node-return-stop-stage.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)));print(r.stdout);assert r.returncode==0,r.stderr

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args();seal()if a.seal else stage()
