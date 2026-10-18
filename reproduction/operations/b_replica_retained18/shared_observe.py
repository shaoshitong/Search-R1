"""Read actual same-card CP64 continuation, without touching the writer."""
import datetime,json,time
from pathlib import Path
from .control import remote,OUT
BASE='/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp64-h1005-20261019-r3'
SCRIPT=r'''
import pathlib,json,time,hashlib
p=pathlib.Path(BASE)
def read(f):return json.loads(f.read_text())if f.exists()else None
def physical(f):
 s=str(f);prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
 return pathlib.Path('/mnt/data/shaoshitong')/s[len(prefix):]if s.startswith(prefix)else pathlib.Path(s)
stage=read(p/'logs/pipeline-status.json');n=max(65,(stage or{}).get('outer',65));root=p/'outers'/('outer-%06d'%n)
progress={str(f.relative_to(p)):read(f)for f in root.rglob('*progress*.json')if f.stat().st_size<15000}
facts=[]
for i in range(65,201):
 j=read(p/'outers'/('outer-%06d'%i)/'complete.json')
 if not j:continue
 ranks=[]
 for r in j['actor']['rank_receipts']:
  f=physical(r['ledger_path']);records=[json.loads(s)for s in f.read_text().splitlines()];restore=next(x for x in records if x['event']=='ACTUAL_RESTORE')
  ranks.append(dict(rank=r['rank'],restored_outer=restore['restored_completed_outer'],adam_before=restore['actual_adam_before'],scheduler_before=restore['restored_scheduler'],writes=sum(x['event']=='ACTUAL_ADAM_WRITE'for x in records),opportunities=sum(x['event']=='OPPORTUNITY'for x in records),ledger_sha_matches=hashlib.sha256(f.read_bytes()).hexdigest()==r['ledger_sha256']))
 facts.append(dict(outer=i,actor=j['actor']['actual_adam_steps_per_rank'],probe=j['probe']['actual_adam_steps_per_rank'],selection_metrics=j['selection_metrics'],phase_timings=j['phase_timings'],ranks=ranks))
active=[]
for f in(p/'actor/outers').glob('outer-%04d-attempt-*/rank-*-updates.jsonl'%n):
 rows=[json.loads(s)for s in f.read_text().splitlines()if s];r=next((x for x in rows if x['event']=='ACTUAL_RESTORE'),None);w=[x for x in rows if x['event']=='ACTUAL_ADAM_WRITE']
 active.append(dict(rank=r['rank']if r else None,restore={k:r.get(k)for k in ('restored_completed_outer','actual_adam_before','restored_scheduler','actual_model_master_dtypes','actual_learning_rates','actual_global_micro')}if r else None,writes=len(w),opportunities=sum(x['event']=='OPPORTUNITY'for x in rows),last_adam=w[-1]['actual_adam_steps']if w else None))
logs=[]
for f in sorted((p/'logs').glob('*.log'),key=lambda f:f.stat().st_mtime,reverse=True)[:4]:
 with f.open('rb')as q:q.seek(max(0,f.stat().st_size-3000));tail=q.read().decode(errors='replace')
 logs.append(dict(path=str(f.relative_to(p)),tail=tail,bytes=f.stat().st_size,mtime=f.stat().st_mtime))
print(json.dumps(dict(observed_unix=time.time(),node='h100-5',stage=stage,complete_outer=(read(p/'latest-joint.json')or{}).get('outer'),facts=facts,active_actor=active,progress=progress,receipt=read(p/'preparation/submit-receipt.json'),cleanup=read(p/'logs/cleanup-status.json'),logs=logs)))
'''

def observe():
 f=OUT/('shared-follow-'+str(time.time_ns())+'-transport.json')
 v=json.loads(remote('BASE='+repr(BASE)+'\n'+SCRIPT,f,target='h100-10',established_controller=True))
 receipt=v.get('receipt');leader=receipt['leader']if receipt else None
 if leader:
  code='WHO='+repr(leader)+'\n'+r'''
import pathlib,json,subprocess,time
p=pathlib.Path('/proc')/str(WHO['pid']);st=(p/'stat').read_text()if p.exists()else None
same=bool(st and int(st.rsplit(')',1)[1].split()[19])==WHO['start_ticks']and st.rsplit(')',1)[1].split()[0]!='Z')
print(json.dumps(dict(observed_unix=time.time(),same_birth=same,stat=st,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
'''
  v['actual_node_health']=json.loads(remote(code,OUT/('shared-follow-health-'+str(time.time_ns())+'.json'),target='h100-5',established_controller=True))
 dest=f.with_name(f.name.replace('-transport','-actual'));dest.write_text(json.dumps(v,ensure_ascii=False,indent=2));v['evidence']=str(dest)
 (OUT/'SHARED_RESUME64_LATEST.json').write_text(json.dumps(v,ensure_ascii=False,indent=2))
 root=OUT.parents[1];s=json.loads((root/'outputs/STATUS.json').read_text());s.update(status='SHARED_NATIVE_CONTINUING'if v.get('actual_node_health',{}).get('same_birth')and not v['cleanup']else'SHARED_NATIVE_EXITED_NEEDS_DIAGNOSIS',actual_training_running=bool(v.get('actual_node_health',{}).get('same_birth'))and not bool(v['cleanup']),actual_complete_outer=v['complete_outer'],current_stage=(v['stage']or{}).get('stage'),node='h100-5',ip='10.33.0.5',world_size=8,resume_job_id=None,cancelled_pending_job=3674,base=receipt['command'][-1]if receipt else None,source_commit=receipt['source_commit']if receipt else None,native_leader=leader,current_evidence=str(dest),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (root/'outputs/STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
 for target in (OUT,root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018')):
  (target/'SHARED_RESUME64_STATUS.json').write_text(json.dumps(v,ensure_ascii=False,indent=2))
 print(json.dumps(dict(time=v['observed_unix'],complete=v['complete_outer'],stage=v['stage'],active=[(r['rank'],r['writes'],r['last_adam'])for r in v['active_actor']],progress={k:r.get('completed_trajectories',r.get('scored_trajectories'))for k,r in v['progress'].items()},health=v.get('actual_node_health'),cleanup=v['cleanup']),ensure_ascii=False),flush=True)
 return v

if __name__=='__main__':observe()
