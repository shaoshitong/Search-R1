"""Publish only this owner's terminal audit artifacts without touching shared HEAD/index."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile
R=Path(__file__).resolve().parents[4]
Q='reproduction/reports/qwen7b-algorithm-improvement-20261021/'
JOBS=[
 ('codex/large-count-smallcp-audit-20261021','7bd62da5f1bda951279ba6864b169ec433603c94','Record actual allocation-only independent small CP1 acceptance',[
 Q+'large-count-smallCP1-independent/actual-real-CP1-acceptance.json',Q+'large-count-smallCP1-independent/FINAL.md']),
 ('codex/large-count-independent-audit-20261021','278ccb2d4afc83e732c7224ded2be3bd550772b8','Record all12 actual own-G5 Count audits and fixed-half internal screen',[
 Q+'large-count-independent-audit/actual-all12-independent.json',Q+'large-count-independent-audit/actual-final-small-readback.json',Q+'large-count-independent-audit/FINAL.json',Q+'large-count-independent-audit/FINAL.md',Q+'large-count-independent-audit/read-final-existing-small-receipts.py',Q+'large-count-independent-audit/publish-terminal-audit-branches.py']),
 ('codex/frozenE1-head-family-20261021','66ea3952e886fddd73ab77d6da0af133dbe89e3c','Record real shared cache CP0 epoch0 metadata readiness for declared36 family',[
 Q+'frozenE1-head-objective-family/actual-input-ready.json',Q+'frozenE1-head-objective-family/ACTUAL-INPUT-READY.md'])]
def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=R,env=env,text=True).strip()
results=[]
for branch,parent,message,files in JOBS:
 assert git('rev-parse','refs/heads/'+branch)==parent,'Unexpected own branch advance'
 remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==parent
 with tempfile.TemporaryDirectory(prefix='audit-isolated-index-')as tmp:
  env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',parent,env=env)
  for name in files:
   p=R/name;assert p.is_file();oid=git('hash-object','-w','--',name);git('update-index','--add','--cacheinfo','100644',oid,name,env=env)
  tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,'-m',message,env=env)
  git('update-ref','refs/heads/'+branch,commit,parent)
  subprocess.run(['git','push','origin',commit+':refs/heads/'+branch],cwd=R,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  assert git('ls-remote','origin','refs/heads/'+branch).split()[0]==commit
  refs=[]
  for name in files:
   raw=(R/name).read_bytes();stored=subprocess.check_output(['git','show',commit+':'+name],cwd=R);assert stored==raw
   refs.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
  results.append(dict(branch=branch,source_commit=commit,remote_SHA=commit,files=refs,shared_HEAD_or_index_modified=False))
out=R.parents[1]/'outputs/qwen7b-warmup-20261019/reports/terminal-count-audit-publications.json';out.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
print(json.dumps(results,sort_keys=True))
