from pathlib import Path
import subprocess,hashlib,json,os,tempfile,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019';branch='codex/prefix20-chosen-logprob-availability-20261021';parent='790d31bf3e58c330b90c2a9b738d58ab3198d164';dest=Path('reports/qwen7b-probe-warmup-20261019/prefix-logprob-availability');files={}
for name in ('prefix-logprob-source-publication.json','prefix-logprob-availability-actual.json','prefix-logprob-final-delivery.json','prefix-logprob-availability-result.md'):files[dest/'actual'/name]=O/'reports'/name
for name in ('operate-prefix-logprob-availability.py','finish-prefix-logprob-report.py','backup-prefix-logprob-final.py'):files[dest/'operators'/name]=O/name
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');cached=git('diff','--cached','--name-only');records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',parent,env=env)
 for rel,source in files.items():
  data=source.read_bytes();(R/rel).parent.mkdir(parents=True,exist_ok=True);(R/rel).write_bytes(data);blob=git('hash-object','-w','--stdin',input=data);git('update-index','--add','--cacheinfo','100644,'+blob+','+str(rel),env=env);records[str(rel)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,input=b'Archive finite actor Prefix20 chosen-logprob availability and causal alignment evidence\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for rel,v in records.items():
 data=subprocess.check_output(['git','show',commit+':'+rel],cwd=R);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==cached
out=dict(branch=branch,commit=commit,remote_sha=remote,files=records,shared_HEAD_and_index_unchanged=True,time=time.time());(O/'reports/prefix-logprob-final-publication.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'}))
