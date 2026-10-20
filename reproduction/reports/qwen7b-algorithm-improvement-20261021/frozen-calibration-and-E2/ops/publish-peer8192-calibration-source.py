from pathlib import Path
import hashlib,json,os,subprocess,tempfile,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';branch='codex/peer8192-frozen-calibration-scoring-20261021'
files=['reproduction/operations/qwen7b_warmup/'+x for x in ('score_peer_g10_calibration.py','deploy_peer_g10_calibration.py','test_peer_g10_calibration.py','score_e2_samehf_baseline.py','deploy_e2_samehf_baseline.py','test_e2_samehf_baseline.py')]
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');parent=git('rev-parse','refs/heads/'+branch);original_index=git('diff','--cached','--name-only');records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',parent,env=env)
 for f in files:
  p=R/f;data=p.read_bytes();blob=git('hash-object','-w','--stdin',input=data);git('update-index','--add','--cacheinfo','100644,'+blob+','+f,env=env);records[f]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,input=b'Add sealed E2 read-only same qualified HF baseline\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True)
remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for f,r in records.items():
 data=subprocess.check_output(['git','show',commit+':'+f],cwd=R);assert hashlib.sha256(data).hexdigest()==r['sha256'] and len(data)==r['bytes']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==original_index
result=dict(branch=branch,commit=commit,remote_sha=remote,files=records,shared_HEAD_and_index_unchanged=True,verified_epoch=time.time())
p=C/'outputs/qwen7b-warmup-20261019/reports/peer8192-calibration-source-publication.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
