"""Precise isolated source branch; preserve shared HEAD and index."""
from pathlib import Path
import subprocess,hashlib,json,os,tempfile,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';branch='codex/peer-expanded-L26-replay-20261021'
names=('peer_expanded_l26_replay.py','deploy_peer_expanded_l26.py','test_peer_expanded_l26.py')
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');cached=git('diff','--cached','--name-only');records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',head,env=env)
 for name in names:
  rel='reproduction/operations/qwen7b_warmup/'+name;data=(R/rel).read_bytes();blob=git('hash-object','-w','--stdin',input=data);git('update-index','--add','--cacheinfo','100644,'+blob+','+rel,env=env);records[rel]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',head,input=b'Capture only additional21824 frozen NQ G10 layer26 with qualified batch32\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for rel,item in records.items():
 data=subprocess.check_output(['git','show',commit+':'+rel],cwd=R);assert hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==cached
value=dict(branch=branch,commit=commit,remote_sha=remote,files=records,shared_HEAD_and_index_unchanged=True,verified_epoch=time.time());(C/'outputs/qwen7b-warmup-20261019/reports/peer-expanded-L26-source-publication.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in value.items() if k!='files'}))
