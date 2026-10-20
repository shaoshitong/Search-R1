"""Read same once service; failed transport never overwrites last valid snapshot."""
import importlib.util,json,os,subprocess,time
from pathlib import Path
from .launch import BASE,ROOT
REMOTE='BASE='+repr(BASE)+'\n'+'''
import pathlib,json,time,subprocess
b=pathlib.Path(BASE);v={'epoch':time.time(),'base':BASE}
v['unit']=subprocess.check_output(['systemctl','show','sst-q7b-newseed-g5-20261020-r3','-p','ActiveState','-p','SubState','-p','MainPID','-p','Result'],text=True)
for n in ['entry-failure.json','failure.json','controller.json','progress.json','code/complete.json','code/direct-failure.json','retrieval/gpu-index-load-progress.json','retrieval/ready.json','complete.json']:
 p=b/n
 if p.exists():
  x=json.loads(p.read_text());x.pop('snapshots',None);v[n]=x
for i in range(2):
 root=b/'sampling'/('shard-%02d'%i);p=root/'progress.json';completed=0
 if p.exists():x=json.loads(p.read_text());v['shard%02d'%i]=x;completed=x['completed_batches']
 for phase in ['prefix','full']:
  p=root/'batches'/('batch-%06d'%completed)/'outers/outer-000001'/phase/'rank-0-progress.json'
  if p.exists():v['shard%02d_current_%s'%(i,phase)]=json.loads(p.read_text())
for n in ['native','retriever','sampling-00','sampling-01']:
 p=b/'logs'/(n+'.log')
 if p.exists():
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-1600));v[n+'_tail']=f.read().decode(errors='replace')
print(json.dumps(v))
'''
def main():
 for n,p in [('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py'),('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py')]:
  s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);globals()[n]=m
 os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1';x=subprocess.run(route.management_argv(ctl.route('h100-11','/usr/bin/python3 -')),input=REMOTE,text=True,capture_output=True,timeout=50);assert x.returncode==0,x.stderr[-500:];v=json.loads(x.stdout);stamp=int(v['epoch']);p=ROOT/'work'/('qwen7b-resample-actual-%d.json'%stamp);p.write_text(json.dumps(v,indent=2));(ROOT/'work/qwen7b-resample-latest-status.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
if __name__=='__main__':main()
