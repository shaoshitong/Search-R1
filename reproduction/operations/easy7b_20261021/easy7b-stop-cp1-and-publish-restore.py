import json,sys,time,subprocess,os
from pathlib import Path
old=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021');new=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-sparse-cp1-20261021');service=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021');commit='921610632eece86be319098351c21b450fd4a236';src=new/'source'/commit
sys.path.insert(0,str(src))
from reproduction.easy2560_joint4.control import bootstrap,publish,sha
from search_g1.easy2560_rolling_checkpoint import latest_checkpoint
assert os.getuid()==14090 and (old/'job.id').read_text().strip()=='745022'
latest=latest_checkpoint(old,verify=False)
if latest is None:print(json.dumps(dict(status='WAITING_COMPLETE_CP1_POINTER',epoch=time.time())));raise SystemExit(0)
assert latest['step']==1 and latest['run_id']=='232f20e112ac4a978c5aba52a0367195'
updates=[json.loads(x)for x in (old/'logs/actual-updates.jsonl').read_text().splitlines()if x]
assert len(updates)==1 and updates[0]['step']==1 and updates[0]['update_status']=='complete' and updates[0]['actual_optimizer_writes']==20 and updates[0]['full_budget_complete'] is True
bootstrap(new,commit)
ready_path=service/'logs/service-ready.json';ready=json.loads(ready_path.read_text())
external=new/'preparation/external-retriever-receipt.json'
if not external.exists():publish(external,dict(service_ready_path=str(ready_path),service_ready_sha256=sha(ready_path),source_commit=ready['source_commit'],service_job_id='745198',endpoint=ready['endpoint']))
from reproduction.easy2560_joint4.retrieval_proxy import external_binding
bound=external_binding(external)
assert bound['service_job_id']=='745198'
for name in ['diagnostics/stage64-four-vs-two.json','diagnostics/actor-visible-parity/result.json']:
 evidence=json.loads((service/name).read_text())
 assert evidence.get('accepted') is True or evidence.get('all_sampled_actor_visible_observations_exact') is True
authorization=json.loads((src/'reproduction/operations/easy7b_20261021/human-cp1-transition-authorization.json').read_text())
assert len(authorization['records'])==3 and authorization['records'][0]['timestamp']=='2026-10-21T10:28:22.987Z'
stop=old/'preparation/human-stop-after-cp1-for-optimized-external2-20261021.json'
if not stop.exists():
 before=subprocess.run(['squeue','-j','745022','-h','-o','%i|%u|%T|%N'],capture_output=True,text=True,timeout=10)
 assert '745022|jxu957|RUNNING|' in before.stdout
 publish(stop,dict(epoch=time.time(),classification='human_authorized_checkpoint_boundary_transition_NOT_failure_NOT_goal_pause',job='745022',authorization=authorization,latest=latest,actual_complete_outer=updates[0],source_and_operator_commit=commit,source_fresh_publication=src.joinpath('source-publication.json').is_file(),queue_before=before.stdout,next_base=str(new),existing_service_job='745198',no_old3b_restart=True))
 r=subprocess.run(['scancel','745022'],capture_output=True,text=True,timeout=15)
 publish(old/'preparation/cp1-transition-scancel-result.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time(),job='745022'))
 assert r.returncode==0
else:
 r=json.loads((old/'preparation/cp1-transition-scancel-result.json').read_text());assert r['returncode']==0
root=new/'preparation/cp1-source-audit';root.mkdir(exist_ok=True);claim=root/'launch.json'
if claim.exists():print(json.dumps(dict(status='SOURCE_AUDIT_ALREADY_LAUNCHED',claim=json.loads(claim.read_text()))));raise SystemExit(0)
output=new/'preparation/cp1-full-state-source-receipt.json'
code='import sys,runpy; sys.path.insert(0,'+repr(str(src))+'); sys.argv=["restore","--base",'+repr(str(old))+',"--checkpoint","latest","--output",'+repr(str(output))+']; runpy.run_module("search_g1.easy2560_restore",run_name="__main__")'
cmd=['srun','--jobid=745198','--overlap','--exact','--ntasks=1','--cpus-per-task=4','--mem=64G','--gres=none','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4',str(new/'env/bin/python'),'-B','-c',code]
publish(claim,dict(epoch=time.time(),source_commit=commit,command=cmd,output=str(output),source_checkpoint=latest['checkpoint'],no_new_gpu_allocation=True,no_model_reload=True,cpu_only_optimizer_and_rng_inspection=True))
wrapper='import subprocess,json,time,pathlib; r=subprocess.run('+repr(cmd)+'); pathlib.Path('+repr(str(root/'exit.json'))+').write_text(json.dumps(dict(returncode=r.returncode,completed_unix=time.time()))+"\\n")'
with (root/'stdout.log').open('xb')as f:p=subprocess.Popen([str(new/'env/bin/python'),'-B','-c',wrapper],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
publish(root/'process.json',dict(pid=p.pid,epoch=time.time()))
print(json.dumps(dict(status='STOPPED_OWN745022_AFTER_COMPLETE_CP1_AND_LAUNCHED_CPU_FULLSTATE_AUDIT',latest=latest,new_base=str(new),audit_pid=p.pid,external_service='745198')))
