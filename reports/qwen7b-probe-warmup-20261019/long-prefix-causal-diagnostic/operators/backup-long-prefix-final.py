from pathlib import Path
import subprocess,hashlib,json,os,tempfile,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019';branch='codex/peer8192-long-prefix-audit-20261021';parent='9d029bda6e49bcde6da98f8060093bc7470b75a7';dest=Path('reports/qwen7b-probe-warmup-20261019/long-prefix-causal-diagnostic');files={}
for name in ('long-prefix-source-publication.json','long-prefix-audit-source-publication.json','long-prefix-stage-latest.json','long-prefix-prepare-latest.json','long-prefix-start-latest.json','long-prefix-progress-1.json','long-prefix-progress-2.json','long-prefix-G10-progress-1.json','long-prefix-status-latest.json','long-prefix-independent-stage-latest.json','long-prefix-independent-audit-latest.json','long-prefix-final-delivery.json','long-prefix-causal-result.md'):
 files[dest/'actual'/name]=O/'reports'/name
for name in ('publish-long-prefix-diagnostic.py','run-long-prefix-independent-audit.py','finish-long-prefix-report.py','backup-long-prefix-final.py'):files[dest/'operators'/name]=O/name
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');cached=git('diff','--cached','--name-only');records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',parent,env=env)
 for rel,source in files.items():
  data=source.read_bytes();(R/rel).parent.mkdir(parents=True,exist_ok=True);(R/rel).write_bytes(data);blob=git('hash-object','-w','--stdin',input=data);git('update-index','--add','--cacheinfo','100644,'+blob+','+str(rel),env=env);records[str(rel)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,input=b'Archive actual frozen8192 causal horizon results and independent aggregate acceptance\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for rel,v in records.items():
 data=subprocess.check_output(['git','show',commit+':'+rel],cwd=R);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==cached
out=dict(branch=branch,commit=commit,remote_sha=remote,files=records,shared_HEAD_and_index_unchanged=True,time=time.time());(O/'reports/long-prefix-final-publication.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'}))
