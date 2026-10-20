"""Stop only r6 at the authorized 01:50 deadline and seal NAS10 handoff."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
RECORDED_BASE=Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
UNIT='sst-q7b-newseed-g5-20261020-r6.service'
RUN_ID='70188b8e50fd4b45bc420ce58014f46e'

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()

def atomic(path,value):
 tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with tmp.open('w')as f:json.dump(value,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 tmp.replace(path)

def birth(pid):
 try:return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except(OSError,IndexError):return None

def service():
 return subprocess.check_output(['systemctl','show',UNIT,'-p','MainPID','-p','Result','-p','ActiveState','-p','ControlGroup'],text=True)

def cgroup_pids(group):
 root=Path('/sys/fs/cgroup')/group.lstrip('/')
 if not group or not root.exists():return []
 found=set()
 for p in root.rglob('cgroup.procs'):
  try:found.update(int(x)for x in p.read_text().split())
  except FileNotFoundError:pass
 return sorted(found)

def handoff(base,out,result):
 assert result['status']=='USER_NODE_RETURN_SAMPLING_STOP_SEALED'
 marker=dict(status='R6_STOP_CONFIRMED_POOL_HANDOFF',run_id=RUN_ID,seed=20261021,r6_stop_receipt=str(RECORDED_BASE/out.relative_to(base)/'complete.json'),r6_stop_receipt_sha256=digest(out/'complete.json'),r6_cgroup_remaining_pids=result['r6_cgroup_remaining_pids'],epoch=time.time())
 atomic(base/'preparation/pool-handoff.json',marker)

def seal():
 base=Path(os.environ.get('Q7B_DATA_BASE',str(RECORDED_BASE)));config=json.loads((base/'config.json').read_text());assert config['run_id']==RUN_ID and config['seed']==20261021
 out=base/'preparation/node-return-20261021-0150';out.mkdir(parents=True,exist_ok=True)
 if(out/'complete.json').exists():
  result=json.loads((out/'complete.json').read_text());assert result['status']=='USER_NODE_RETURN_SAMPLING_STOP_SEALED';handoff(base,out,result);print(json.dumps(result));return
 before_service=service();props=dict(line.split('=',1)for line in before_service.splitlines()if'='in line);pid=int(props.get('MainPID',0));owned={pid}if pid else set();group=props.get('ControlGroup','');owned.update(cgroup_pids(group));links={}
 for p in Path('/proc').glob('[0-9]*/stat'):
  try:links[int(p.parent.name)]=int(p.read_text().rsplit(')',1)[1].split()[1])
  except(OSError,ValueError,IndexError):pass
 while True:
  new=owned|{p for p,parent in links.items()if parent in owned}
  if new==owned:break
  owned=new
 births={str(p):birth(p)for p in owned};intent=dict(user_instruction='Keep 02:00 node return; stop this sampling at 01:50',unit=UNIT,run_id=RUN_ID,seed=20261021,epoch=time.time(),owned_births=births,service_before=before_service,control_group=group,full_data_remain_on_nas10=True)
 if not(out/'intent.json').exists():atomic(out/'intent.json',intent)
 else:
  previous=json.loads((out/'intent.json').read_text());assert previous['run_id']==RUN_ID;births.update(previous['owned_births'])
 stopped=subprocess.run(['systemctl','stop',UNIT],text=True,capture_output=True,timeout=100)
 for _ in range(20):
  remaining={p:b for p,b in births.items()if b is not None and birth(int(p))==b};groups=cgroup_pids(group)
  if not remaining and not groups:break
  time.sleep(1)
 final_service=service();props=dict(line.split('=',1)for line in final_service.splitlines()if'='in line);terminal=props.get('MainPID')=='0'and props.get('ActiveState')in('inactive','failed')
 import shutil
 controls=[];logs=[];target=out/'root-controls';target.mkdir(exist_ok=True)
 for folder in Path('/root/.local/share/search-g1/qwen7b-newseed-g5').glob('r[0-9]*'):
  if not folder.is_dir():continue
  dest=target/folder.name;dest.mkdir(exist_ok=True)
  for f in folder.iterdir():
   if f.is_file():
    st=f.stat();copy=dest/f.name;shutil.copy2(f,copy);original=digest(f);assert digest(copy)==original;controls.append(dict(original=str(f),original_inode=st.st_ino,bytes=st.st_size,retained=str(copy),sha256=original))
 if not remaining and not groups:
  for p in sorted((base/'logs/r6').glob('*.log')):
   st=p.stat();logs.append(dict(path=str(p),inode=st.st_ino,bytes=st.st_size,sha256=digest(p)))
 rows=[json.loads(p.read_text())for p in(base/'sampling').glob('shard-*/batches/batch-*/complete.json')]
 result=dict(status='USER_NODE_RETURN_SAMPLING_STOP_SEALED'if not remaining and not groups and terminal and stopped.returncode==0 else'STOP_NEEDS_ATTENTION',unit=UNIT,run_id=RUN_ID,seed=20261021,epoch=time.time(),questions=sum(x['questions']for x in rows),trajectories=sum(x['trajectories']for x in rows),complete_batches=len(rows),remaining_original_births=remaining,r6_cgroup_remaining_pids=groups,terminal_logs=logs,root_control_copies=controls,stop_returncode=stopped.returncode,stop_stderr=stopped.stderr,service=final_service,goal_complete=False,shared_data_deleted=False)
 if result['status']=='USER_NODE_RETURN_SAMPLING_STOP_SEALED':atomic(out/'complete.json',result);handoff(base,out,result)
 else:atomic(out/('attention-%d.json'%time.time_ns()),result)
 print(json.dumps(result));assert result['status']=='USER_NODE_RETURN_SAMPLING_STOP_SEALED'

def stage():
 import base64,importlib.util
 from .launch import ROOT
 source=Path(__file__).read_bytes();source_sha=hashlib.sha256(source).hexdigest()
 remote='SOURCE='+repr(base64.b64encode(source).decode())+'\nSHA='+repr(source_sha)+'\n'+r'''
import base64,hashlib,json,subprocess,time
from pathlib import Path
root=Path('/root/.local/share/search-g1/qwen7b-newseed-g5/r6');assert(root/'intent.json').is_file()
p=root/'node-return-stop-intent.json';timer='sst-q7b-before-node-return-r6'
if p.exists():assert subprocess.check_output(['systemctl','is-active',timer+'.timer'],text=True).strip()=='active'
data=base64.b64decode(SOURCE);assert hashlib.sha256(data).hexdigest()==SHA
helper=root/'node_return_stop.py';old=hashlib.sha256(helper.read_bytes()).hexdigest()if helper.exists()else None
if old!=SHA:
 if old:(root/('node_return_stop-'+old+'.py')).write_bytes(helper.read_bytes())
 tmp=root/'node_return_stop.py.tmp';tmp.write_bytes(data);tmp.replace(helper)
wrapper='/root/.local/share/search-g1/source8-fault-recovery/b2d62068695e156dcc90fc42e112ad24c676a1e7531fc26e2c0c6a01dbbfc1c1/retained-fault-paths.sh'
lines=['#!/bin/bash','set -euo pipefail','pid=$(systemctl show sst-q7b-newseed-g5-20261020-r6.service -p MainPID --value)','if [[ "$pid" != 0 ]]; then',' exec nsenter --mount --target "$pid" /usr/bin/python3 '+str(helper)+' --seal','fi','export Q7B_DATA_BASE=/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020','exec unshare --mount --propagation private bash '+wrapper+' h100-11 /usr/bin/python3 '+str(helper)+' --seal','']
script=root/'stop-at-0150.sh';content=chr(10).join(lines).encode();tmp=root/'stop-at-0150.sh.tmp';tmp.write_bytes(content);subprocess.run(['bash','-n',str(tmp)],check=True)
if script.exists()and script.read_bytes()!=content:
 oldscript=hashlib.sha256(script.read_bytes()).hexdigest();(root/('stop-at-0150-'+oldscript+'.sh')).write_bytes(script.read_bytes())
tmp.replace(script)
v=dict(epoch=time.time(),deadline='2026-10-21 01:50:00 Asia/Shanghai',only_unit='sst-q7b-newseed-g5-20261020-r6.service',source_sha256=SHA,wrapper_sha256=hashlib.sha256(content).hexdigest(),keep_existing_0200_wipe=True)
if p.exists():
 result=dict(existing_timer_not_replayed=True,old_source_sha256=old,current_source_sha256=SHA,wrapper_sha256=v['wrapper_sha256'],epoch=time.time());(root/('node-return-stop-refresh-'+SHA+'.json')).write_text(json.dumps(result));print(json.dumps(result));raise SystemExit(0)
with p.open('x')as f:json.dump(v,f)
r=subprocess.run(['systemd-run','--unit='+timer,'--on-calendar='+v['deadline'],'--timer-property=AccuracySec=1s','/bin/bash',str(script)],text=True,capture_output=True);result=dict(intent=v,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,timer=subprocess.check_output(['systemctl','show',timer+'.timer','-p','NextElapseUSecRealtime','-p','ActiveState'],text=True));(root/'node-return-stop-receipt.json').write_text(json.dumps(result));print(json.dumps(result));assert r.returncode==0
'''
 for name,path in[('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py'),('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py')]:
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);globals()[name]=module
 os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1';r=subprocess.run(route.management_argv(ctl.route('h100-11','/usr/bin/python3 -')),input=remote,text=True,capture_output=True,timeout=60);(ROOT/'work/qwen7b-node-return-stop-stage.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)));print(r.stdout);assert r.returncode==0,r.stderr

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args();seal()if a.seal else stage()
