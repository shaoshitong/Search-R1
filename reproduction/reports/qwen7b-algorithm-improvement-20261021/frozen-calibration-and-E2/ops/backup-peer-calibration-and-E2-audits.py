"""Publish owned code/ops/small aggregate audits through a separate Git index."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,time
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019';P=O/'reports';branch='codex/peer8192-frozen-calibration-scoring-20261021'
cal=json.loads((P/'peer8192-calibration-independent-final-audit.json').read_text());e2=json.loads((P/'E2-same-HF-independent-final-audit.json').read_text())
summary=dict(status='FROZEN_PEER_G10_CALIBRATION_SCORES_AND_E2_SAME_HF_BASELINE_FULLY_VERIFIED',
 source_branch=branch,scorer_runtime_commit='a0f41cb45daa41ed811577037d950f4ff12dfe44',E2_runtime_commit='6fc0ac0894c093632f5c1dbdeaf06c2f3ada668e',pinned_model_reader_runtime_commit='d8f74420aa09fa56e8ec59207c6d1453fa86f95e',
 primary_job=dict(job_id=cal['job_id'],sacct=cal['sacct'],terminal_normal_exit=cal['terminal_normal_exit'],config=cal['config'],complete=cal['complete'],groups=cal['groups'],CPU_compatible_inputs=cal['CPU_compatible_inputs'],identities=cal['identities'],models=cal['models'],scope='seen training NQ8192 G10 calibration, not unseen performance'),
 E2_job=e2,same_HF_fair_grid=e2['comparison_same_HF_frozen2048'],
 decision=dict(best_BA_model='E2',best_BA=e2['comparison_same_HF_frozen2048']['E2']['BA'],best_Top512_model='E4',best_Top512=196,peer_new_data_recipe_beats_both_primary_gates=False,larger_goal_solved=False,untouched_final_confirmation_started=False),
 failures=dict(initial_job743412='FAILED1:0 before any scores due to legacy SHA-only model_config missing bytes',fix='Preserve mandatory SHA; validate bytes only when present; actual12 metadata refs preflight passed; immutable v2 only',no_live_hotfix=True),
 tests=dict(primary_cpu_regressions=8,E2_cpu_regressions=3,actual_export_execution_and_bash_n=True,actual_metadata_preflights=True),
 actor_updates=0,probe_updates=0,optimizer_created=False,source_weights_Adam_unchanged=True,final_pool_read=False,model_or_feature_payload_local=False,observed_epoch=time.time())
(P/'peer8192-calibration-and-E2-final-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
ops=['publish-peer8192-calibration-source.py','peer8192-calibration-external-refs-readonly.py','peer8192-calibration-input-refs-readonly.py','peer8192-calibration-v2-metadata-preflight.py','peer8192-calibration-independent-audit.py','poll-peer8192-calibration.py','E2-same-HF-metadata-preflight.py','E2-same-HF-independent-audit.py','backup-peer-calibration-and-E2-audits.py','frozen-score-audit-refs-readonly.py']
reports=sorted({p for pattern in ('peer8192-calibration-*.json','E2-same-HF-*.json') for p in P.glob(pattern) if 'final-publication' not in p.name})
files={O/n:'reproduction/reports/qwen7b-algorithm-improvement-20261021/frozen-calibration-and-E2/ops/'+n for n in ops};files.update({p:'reproduction/reports/qwen7b-algorithm-improvement-20261021/frozen-calibration-and-E2/summaries/'+p.name for p in reports})
def git(*args,env=None,input=None):return subprocess.check_output(['git',*args],cwd=R,env=env,input=input).decode().strip()
head=git('rev-parse','HEAD');cached=git('diff','--cached','--name-only');parent=git('rev-parse','refs/heads/'+branch);records={}
with tempfile.TemporaryDirectory() as tmp:
 env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/'index'));git('read-tree',parent,env=env)
 for source,dest in files.items():
  payload=source.read_bytes();assert len(payload)<256*1024,source;oid=git('hash-object','-w','--stdin',input=payload);git('update-index','--add','--cacheinfo','100644,'+oid+','+dest,env=env);records[dest]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
 tree=git('write-tree',env=env);commit=git('commit-tree',tree,'-p',parent,input=b'Preserve actual frozen G10 scores, metadata fix and E2 fair-HF audits\n');git('update-ref','refs/heads/'+branch,commit)
subprocess.run(['git','push','origin',branch],cwd=R,check=True);remote=git('ls-remote','origin','refs/heads/'+branch).split()[0];assert remote==commit
for name,item in records.items():
 payload=subprocess.check_output(['git','show',commit+':'+name],cwd=R);assert hashlib.sha256(payload).hexdigest()==item['sha256'] and len(payload)==item['bytes']
assert git('rev-parse','HEAD')==head and git('diff','--cached','--name-only')==cached
r=dict(branch=branch,commit=commit,remote_sha=remote,owned_audit_files=len(records),bytes=sum(x['bytes'] for x in records.values()),file_manifest=records,shared_HEAD_and_index_unchanged=True,verified_epoch=time.time());(P/'peer8192-calibration-final-publication.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='file_manifest'},sort_keys=True))
