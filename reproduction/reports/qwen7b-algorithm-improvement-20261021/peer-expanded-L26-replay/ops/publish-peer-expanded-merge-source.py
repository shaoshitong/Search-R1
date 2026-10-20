"""Isolated CPU merger publication atop fixed capture source, no main edits."""
from pathlib import Path
import subprocess,tempfile,os,hashlib,json,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019';branch='codex/peer-expanded-L26-merge-20261021';prefix='reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-expanded-L26-replay/'
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');cached=git('diff','--cached','--name-only')
try:parent=git('rev-parse','refs/heads/'+branch)
except subprocess.CalledProcessError:parent='bea03d5f1d908f357c368684b9e8c086ca0e687b'
files={R/('reproduction/operations/qwen7b_warmup/'+n):'reproduction/operations/qwen7b_warmup/'+n for n in ('peer_expanded_l26_merge.py','test_peer_expanded_l26_merge.py')}
files.update({O/n:prefix+'ops/'+n for n in ('deploy-peer-expanded-merge.py','publish-peer-expanded-merge-source.py','publish-peer-expanded-L26-source.py','poll-peer-expanded-L26.py','inspect-expanded-shard0.py','inspect-expanded-merge-cpu.py','peer-expanded-L26-independent-audit.py','fetch-peer-expanded-final-audit.py','read-peer-expanded-final-audit-remote.py') if (O/n).exists()})
files.update({p:prefix+'summaries/'+p.name for p in (O/'reports').glob('peer-expanded-L26-*.json') if 'merge-source-publication' not in p.name})
records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',parent,env=env)
 for p,name in files.items():
  data=p.read_bytes();assert len(data)<256*1024,p;blob=git('hash-object','-w','--stdin',input=data);git('update-index','--add','--cacheinfo','100644,'+blob+','+name,env=env);records[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,input=b'Preserve additional21824 qualified replay launch and independent CPU ownEM merger\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for name,r in records.items():
 data=subprocess.check_output(['git','show',commit+':'+name],cwd=R);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==cached
v=dict(branch=branch,commit=commit,remote_sha=remote,files=records,shared_HEAD_and_index_unchanged=True,verified_epoch=time.time());(O/'reports/peer-expanded-L26-merge-source-publication.json').write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in v.items() if k!='files'}))
