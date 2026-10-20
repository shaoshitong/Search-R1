"""Snapshot only owned within1 code/ops/small aggregates, separate Git index."""
from pathlib import Path
import json,hashlib,os,subprocess,tempfile,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019';P=O/'reports';branch='codex/peer8192-within1-matched256-20261021';prefix='reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-within1-matched256/'
ops=[O/n for n in ('publish-peer-within1-source.py','read-peer-control-runtime-publication.py','poll-peer-within1.py','peer-within1-first-independent-audit.py','peer-within1-final-independent-audit.py','fetch-peer-within1-final-audit.py','read-peer-within1-final-audit-remote.py','backup-peer-within1-audits.py')]
reports=sorted(p for p in P.glob('peer-within1-*.json') if 'backup-publication' not in p.name);reports.append(P/'peer-control-D8-runtime-source-publication.json');files={p:prefix+'ops/'+p.name for p in ops if p.exists()};files.update({p:prefix+'summaries/'+p.name for p in reports if p.exists()})
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');cached=git('diff','--cached','--name-only');parent=git('rev-parse','refs/heads/'+branch);records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',parent,env=env)
 for p,name in files.items():
  data=p.read_bytes();assert len(data)<256*1024,p;blob=git('hash-object','-w','--stdin',input=data);git('update-index','--add','--cacheinfo','100644,'+blob+','+name,env=env);records[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,input=b'Preserve actual matched within1 progress and independent checkpoint audits\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for name,v in records.items():
 data=subprocess.check_output(['git','show',commit+':'+name],cwd=R);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==cached
r=dict(branch=branch,commit=commit,remote_sha=remote,files=records,owned_files=len(records),bytes=sum(v['bytes'] for v in records.values()),shared_HEAD_and_index_unchanged=True,verified_epoch=time.time());(P/'peer-within1-backup-publication.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='files'}))
