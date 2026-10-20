from pathlib import Path
import subprocess,tempfile,os,hashlib,json,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019';branch='codex/frozen-readout-contribution-20261021';prefix='reproduction/reports/qwen7b-algorithm-improvement-20261021/frozen-readout-contribution/'
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');cached=git('diff','--cached','--name-only');parent=git('rev-parse','refs/heads/'+branch)
files={O/n:prefix+'ops/'+n for n in ('publish-readout-diagnostic-source.py','read-readout-prereqs.py','run-readout-cpu-tests.py','readout-independent-audit.py','backup-readout-audits.py','fetch-readout-final-audit.py','read-readout-final-audit-remote.py') if (O/n).exists()}
files.update({p:prefix+'summaries/'+p.name for p in (O/'reports').glob('readout-contribution-*') if p.suffix in ('.json','.md') and 'backup-publication' not in p.name});records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',parent,env=env)
 for p,name in files.items():
  data=p.read_bytes();assert len(data)<256*1024,p;blob=git('hash-object','-w','--stdin',input=data);git('update-index','--add','--cacheinfo','100644,'+blob+','+name,env=env);records[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,input=b'Preserve frozen native readout contribution actual run and independent audit\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for name,r in records.items():
 data=subprocess.check_output(['git','show',commit+':'+name],cwd=R);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==cached
v=dict(branch=branch,commit=commit,remote_sha=remote,files=records,shared_HEAD_and_index_unchanged=True,time=time.time());(O/'reports/readout-contribution-backup-publication.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in v.items() if k!='files'}))
