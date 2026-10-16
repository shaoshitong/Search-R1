import sys,os,json,time,hashlib,shlex
from pathlib import Path
sys.path.insert(0,'work/tree-speed-repair-20261008')
import ssh_retained18 as ssh
os.environ['TREE_MGMT_CONTROL_PATH']='/tmp/tree-mgmt-3fw9a1ab/control'
source=r'''import json,time,subprocess,hashlib
from pathlib import Path
u='sst-tree-nested-final-eval-tp1-778b2b43-20261016.service'
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs')/u.removesuffix('.service')
s=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:778b2b43dad90fc0c1c5d95d77c2326286011b73/output/final-eval-fast/scores')
def tail(p,n=9000):
 if not p.is_file():return None
 with p.open('rb') as f:
  f.seek(0,2);size=f.tell();f.seek(max(0,size-n));b=f.read()
 return dict(bytes=size,tail=b.decode(errors='replace'))
def small(p):
 if not p.is_file():return None
 b=p.read_bytes()
 return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),value=json.loads(b))
a=subprocess.run(['systemctl','show',u,'--property=LoadState,ActiveState,SubState,MainPID,InvocationID,Result,ExecMainStatus,ControlGroup,CPUUsageNSec,MemoryCurrent'],capture_output=True,text=True)
x=dict(epoch=time.time(),systemd=a.stdout,systemctl_returncode=a.returncode,files={},logs={})
for name in ['allocation-identity.json','worker-gpu-mapping.json','run-result.json']:x['files'][name]=small(r/name)
for name in ['retriever.log','eval.log','guard.log','resource-samples.jsonl']:x['logs'][name]=tail(r/name)
comm=[]
for p in sorted((s/'batches').glob('*.done.json')):
 b=p.read_bytes();v=json.loads(b);data_path=p.with_name(p.name.removesuffix('.done.json')+'.json');raw=data_path.read_bytes();data=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==v['sha256'] and len(data['rows'])==v['count'];comm.append(dict(name=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),value=v,rank=data['rank'],cost=data['cost'],identity=data['identity'],execution=data['execution'],first_index=data['rows'][0]['index'],last_index=data['rows'][-1]['index'],committed_mtime_epoch=p.stat().st_mtime))
x['committed_batches']=comm
for p in sorted(s.glob('*provenance.json')):x['files'][p.name]=small(p)
for n in ['tokenizer-hf200-reload-proof.json','final-metrics.json','final-metrics.done.json']:x['files'][n]=small(s/n)
x['journal']=subprocess.run(['journalctl','-u',u,'-n','35','--no-pager','-o','short-iso'],capture_output=True,text=True).stdout[-18000:]
print(json.dumps(x,ensure_ascii=False))'''
t=time.time();response=ssh.run('python3 -c '+shlex.quote(source),timeout=90)
out=Path('outputs/tree-nested-branch-handoff-20261013')/sys.argv[1]
if out.exists():raise RuntimeError('immutable observation exists')
wrapper=dict(local_observed_epoch=time.time(),wall_seconds=time.time()-t,returncode=response.returncode,stderr=response.stderr)
if response.returncode==0:wrapper['remote']=json.loads(response.stdout)
else:wrapper['stdout']=response.stdout
out.write_text(json.dumps(wrapper,ensure_ascii=False,indent=2)+'\n')
r=wrapper.get('remote',{});print(json.dumps(dict(returncode=wrapper['returncode'],epoch=r.get('epoch'),systemd=r.get('systemd'),batches=len(r.get('committed_batches',[])),rows=sum(v['value']['count'] for v in r.get('committed_batches',[])),eval_tail=(r.get('logs',{}).get('eval.log') or {}).get('tail','')[-5000:],result=r.get('files',{}).get('run-result.json')),ensure_ascii=False))
