"""Back up owned read-only collectors and compact result summaries only.

Use an isolated Git index; never alter another experiment's checkout or index.
"""
from pathlib import Path
import subprocess,tempfile,os,json,hashlib,re,datetime
D=Path(__file__).resolve().parent
repo=D.parent.parent/'work/search-e1-publish'
branch='codex/session-results-audit-20261015'
base=['git','-C',str(repo)]
def run(args,**kwargs):return subprocess.run(base+args,check=True,capture_output=True,**kwargs).stdout
files={p.name:p.read_bytes() for p in D.glob('*.py')}
for name in ['all-seven-results.json','g1-audit.json','tree-e1-b-audit.json','gigpo-co-audit.json','topd-audit.json']:
 files[name]=(D/name).read_bytes()
for name,raw in files.items():assert not re.search(rb'hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----',raw),name
target=branch if run(['ls-remote','origin','refs/heads/'+branch]).strip() else 'codex/reward-selection-figures-20261014'
run(['fetch','origin',target]);parent=run(['rev-parse','FETCH_HEAD']).decode().strip()
prefix='experiments/search_e1/control/session_results_audit_20261015/'
with tempfile.TemporaryDirectory(prefix='session-audit-index-') as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));run(['read-tree',parent],env=env)
 for name,raw in files.items():
  blob=run(['hash-object','-w','--stdin'],input=raw).decode().strip()
  run(['update-index','--add','--cacheinfo','100644',blob,prefix+name],env=env)
 tree=run(['write-tree'],env=env).decode().strip()
 commit=run(['commit-tree',tree,'-p',parent],input=b'Archive read-only seven-session experiment result audit\n').decode().strip()
run(['push','origin',commit+':refs/heads/'+branch])
remote=run(['ls-remote','origin','refs/heads/'+branch]).decode().split()[0];assert remote==commit
run(['fetch','origin',branch]);assert run(['rev-parse','FETCH_HEAD']).decode().strip()==commit
for name,raw in files.items():assert run(['show',commit+':'+prefix+name])==raw
receipt={'session':'01a101e0-90d3-7731-bae9-58c8f1fdf7e2','source_commit':commit,'remote_sha':remote,'branch':branch,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkout_unchanged':True,'files':[{'path':prefix+n,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} for n,raw in files.items()]}
(D/'publication-verified.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({k:receipt[k] for k in ['source_commit','remote_sha','branch','verified_utc']}))
