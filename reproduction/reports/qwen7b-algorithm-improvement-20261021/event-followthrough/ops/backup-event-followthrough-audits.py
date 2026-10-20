"""Back up only owned code/small aggregate audits without switching the shared checkout."""
import datetime,hashlib,json,os,pathlib,subprocess,tempfile
C=pathlib.Path(__file__).resolve().parents[2]
R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019'
PREFIX='reproduction/reports/qwen7b-algorithm-improvement-20261021/event-followthrough'
OPS=['training-enrichment-select-control.py','training-enrichment-independent-audit.py',
     'hard-random477-readonly-status.py','poll-candidate-and-layer-progress.py',
     'hard-random477-independent-first-audit.py',
     'build-hard-random-followthrough-report.py',
     'four-probe-jobs-readonly-status.py','g10-first-updates-independent-audit.py',
     'g10-prefix-native-first-batch-audit.py','four-probe-final-independent-audit.py',
     'poll-four-probe-jobs.py',
     'multilayer-readonly-checkpoint-status.py','multilayer-epoch1-independent-audit.py',
     'multilayer-final-independent-audit.py','frozen-validation-independent-audit.py',
     'backup-event-followthrough-audits.py']
REPORTS=['training-enrichment-independent-audit-final.json','training-candidate-followthrough-status-24.json',
    'multilayer-final-independent-audit-final.json','multilayer-first-epoch-followthrough-final.json',
    'multilayer-epoch1-independent-audit-final.json','frozen-validation-independent-audit-final.json',
    'frozen-validation-followthrough-final.json','hard-random477-actual-7.json',
    'hard-random477-independent-first-audit-final.json','hard-random477-independent-cp32-audit.json',
    'hard-random477-followthrough-final.json','g10-first-updates-independent-audit-final.json',
    'g10-prefix-native-first-batch-audit-final.json','four-probe-jobs-actual-22.json']
def git(*args,input=None,env=None):
 return subprocess.check_output(['git',*args],cwd=R,input=input,env=env).decode().strip()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
branch='codex/qwen7b-event-followthrough-audits-'+stamp
base=git('rev-parse','HEAD');files={}
for category,names in [('ops',OPS),('summaries',REPORTS)]:
 for name in names:
  p=(O/name) if category=='ops' else (O/'reports'/name)
  raw=p.read_bytes()
  assert p.suffix in ('.py','.json') and len(raw)<100000
  assert (b'h'+b'f_') not in raw and (b'BEGIN '+b'OPENSSH PRIVATE KEY') not in raw
  key=PREFIX+'/'+category+'/'+name
  files[key]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),raw=raw)
for name in ('replay_batched_context.py','test_replay_batched_context.py'):
 p=R/'reproduction/operations/qwen7b_warmup'/name;raw=p.read_bytes()
 key='reproduction/operations/qwen7b_warmup/'+name
 files[key]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),raw=raw)
manifest=dict(status='OWNED_CODE_AND_SMALL_AGGREGATE_AUDIT_BACKUP',created_utc=stamp,
    base_commit=base,branch=branch,shared_checkout_or_index_modified=False,
    model_checkpoint_optimizer_raw_predictions_or_credentials_included=False,
    files={k:{a:b for a,b in v.items() if a!='raw'} for k,v in files.items()})
manifest_path=PREFIX+'/manifest.json'
with tempfile.TemporaryDirectory(prefix='q7b-audit-index-') as td:
 env=dict(os.environ,GIT_INDEX_FILE=str(pathlib.Path(td)/'index'))
 git('read-tree',base,env=env)
 for path,item in files.items():
  blob=git('hash-object','-w','--stdin',input=item['raw']);git('update-index','--add','--cacheinfo','100644,'+blob+','+path,env=env)
 raw=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();blob=git('hash-object','-w','--stdin',input=raw)
 git('update-index','--add','--cacheinfo','100644,'+blob+','+manifest_path,env=env)
 tree=git('write-tree',env=env)
 commit=git('commit-tree',tree,'-p',base,input=b'Back up independently verified probe evaluation and hard-selection audits\n')
 git('update-ref','refs/heads/'+branch,commit,'0'*40)
 git('push','origin','refs/heads/'+branch+':refs/heads/'+branch)
 remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
 for path,item in files.items():
  saved=subprocess.check_output(['git','show',commit+':'+path],cwd=R)
  assert hashlib.sha256(saved).hexdigest()==item['sha256'] and len(saved)==item['bytes']
 receipt=dict(manifest,commit=commit,remote_sha=remote,manifest_sha256=hashlib.sha256(raw).hexdigest(),files_verified=True)
 dest=O/'reports'/'event-followthrough-publication.json';dest.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(status=receipt['status'],branch=branch,commit=commit,remote_sha=remote,
    files=len(files),total_bytes=sum(x['bytes'] for x in files.values()),files_verified=True,
    shared_checkout_or_index_modified=False,receipt=str(dest))))
