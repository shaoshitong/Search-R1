from pathlib import Path
import subprocess,hashlib,json,os,tempfile,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';branch='codex/frozen-readout-contribution-20261021'
names=('score_readout_contribution.py','deploy_readout_contribution.py','test_readout_contribution.py','score_peer_g10_calibration.py','deploy_peer_g10_calibration.py')
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');cached=git('diff','--cached','--name-only');records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',head,env=env)
 for name in names:
  rel='reproduction/operations/qwen7b_warmup/'+name;data=(R/rel).read_bytes();blob=git('hash-object','-w','--stdin',input=data);git('update-index','--add','--cacheinfo','100644,'+blob+','+rel,env=env);records[rel]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',head,input=b'Add frozen question-prefix contribution diagnostic without changing production forward\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for rel,v in records.items():
 data=subprocess.check_output(['git','show',commit+':'+rel],cwd=R);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==cached
out=dict(branch=branch,commit=commit,remote_sha=remote,files=records,shared_HEAD_and_index_unchanged=True,time=time.time());(C/'outputs/qwen7b-warmup-20261019/reports/readout-contribution-source-publication.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'}))
