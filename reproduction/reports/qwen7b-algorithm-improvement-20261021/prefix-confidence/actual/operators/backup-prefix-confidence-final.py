from pathlib import Path
import subprocess,hashlib,json,os,tempfile,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019';branch='codex/prefix-confidence-incremental-20261021';parent='e11fe2b1cc01977dcb0bde57490967e2445ce98f';dest=Path('reproduction/reports/qwen7b-algorithm-improvement-20261021/prefix-confidence/actual');names=('prefix-confidence-data-source-publication.json','prefix-confidence-data-stage-latest.json','prefix-confidence-data-prepare-latest.json','prefix-confidence-data-start-latest.json','prefix-confidence-data-status-latest.json','prefix-confidence-fit-source-publication.json','prefix-confidence-fit-v2-source-publication.json','prefix-confidence-fit-stage-latest.json','prefix-confidence-fit-start-latest.json','prefix-confidence-fit-status-latest.json','prefix-confidence-small-configs.json','prefix-confidence-final-delivery.json','prefix-confidence-result.md','prefix-confidence-independent-final.json');files={dest/n:O/'reports'/n for n in names};ops=('prefix-confidence-input-schema.py','prefix-confidence-small-configs.py','prefix-confidence-gate-patch.py','backup-prefix-confidence-final.py')
for n in ops:files[dest/'operators'/n]=O/n
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');cached=git('diff','--cached','--name-only');records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',parent,env=env)
 for rel,source in files.items():
  data=source.read_bytes();assert len(data)<150000,'Only small audit/config summaries, never real model/scalar tensor or full rows';(R/rel).parent.mkdir(parents=True,exist_ok=True);(R/rel).write_bytes(data);blob=git('hash-object','-w','--stdin',input=data);git('update-index','--add','--cacheinfo','100644,'+blob+','+str(rel),env=env);records[str(rel)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,input=b'Archive real native-prefix confidence data quality and matched2/full8 negative terminal evidence\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for rel,v in records.items():
 data=subprocess.check_output(['git','show',commit+':'+rel],cwd=R);assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==cached
out=dict(branch=branch,commit=commit,remote_sha=remote,files=records,shared_HEAD_and_index_unchanged=True,real_parameters_normalizer_optimizer_RNG_and_rows_remote_only=True,time=time.time());(O/'reports/prefix-confidence-final-publication.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'}))
