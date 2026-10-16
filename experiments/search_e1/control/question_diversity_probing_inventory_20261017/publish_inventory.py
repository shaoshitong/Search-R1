"""Publish owned audit code and compact catalog through an isolated Git index."""
from pathlib import Path
import subprocess,tempfile,os,json,hashlib,re,datetime
D=Path(__file__).resolve().parent
repo=D.parent.parent/'work/search-e1-publish'
branch='codex/question-diversity-probe-inventory-20261017'
def run(args,**kw):
 return subprocess.run(['git','-C',str(repo)]+args,check=True,capture_output=True,**kw).stdout
names=['scan_full_pinned_history.py','condense_chronology.py','collect_actual_metadata.py','build_inventory.py','publish_inventory.py',
 'REPORT.md','experiments.json','execution-attempts.json','inventory-summary.json','canonical-handle-coverage.json',
 'full-history-scan-audit.json','cited-source-manifest.json','frozen-local-key-facts.json','pinned-sessions.json']
files={n:(D/n).read_bytes() for n in names}
for n,b in files.items():
 assert not re.search(rb'hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----',b),n
remote=run(['ls-remote','origin','refs/heads/'+branch]).decode().strip()
parentbranch=branch if remote else 'codex/session-results-audit-20261017'
run(['fetch','origin',parentbranch]);parent=run(['rev-parse','FETCH_HEAD']).decode().strip()
prefix='experiments/search_e1/control/question_diversity_probing_inventory_20261017/'
with tempfile.TemporaryDirectory(prefix='qdp-audit-index-') as temp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'))
 run(['read-tree',parent],env=env)
 for n,b in files.items():
  blob=run(['hash-object','-w','--stdin'],input=b).decode().strip()
  run(['update-index','--add','--cacheinfo','100644',blob,prefix+n],env=env)
 tree=run(['write-tree'],env=env).decode().strip()
 commit=run(['commit-tree',tree,'-p',parent],input=b'Inventory QDP warmups, settings, failed starts and recovery histories\n').decode().strip()
run(['push','origin',commit+':refs/heads/'+branch])
sha=run(['ls-remote','origin','refs/heads/'+branch]).decode().split()[0];assert sha==commit
run(['fetch','origin',branch]);assert run(['rev-parse','FETCH_HEAD']).decode().strip()==commit
for n,b in files.items():assert run(['show',commit+':'+prefix+n])==b
receipt={'session_id':'01a101e0-90d3-7731-bae9-58c8f1fdf7e2','branch':branch,'commit':commit,'remote_sha':sha,
 'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkout_and_index_unchanged':True,
 'files':[{'path':prefix+n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()]}
(D/'publication-verified.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['branch','commit','remote_sha','verified_utc']}))
