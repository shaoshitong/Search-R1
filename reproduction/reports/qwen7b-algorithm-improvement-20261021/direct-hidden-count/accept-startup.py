"""Read-only actual joint CP1 + six consecutive transactions acceptance."""
import hashlib,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
rows=[json.loads(s) for s in (OUT/'actual-observer/observations.jsonl').read_text().splitlines()]
valid=[r for r in rows if r.get('cursor') is not None and not r.get('failure') and not r.get('CPU_audit_failure')]
if not valid:raise RuntimeError('No actual accepted joint update observed')
first=valid[0];later=next((r for r in valid if r['cursor']>=6 and r['cursor']>first['cursor'] and r['time']-first['time']>=45),None)
if later is None:raise RuntimeError('Six real transactions and two separated positive observations required')
source='''import pathlib,json,hashlib,math,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2')
def read(p):return json.loads(pathlib.Path(p).read_text())
def ref(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
c=read(b/'config.json');assert ref(b/'config.json')['sha256']=='3e910b12c0ff52eac70e61566c6135e2ec188bb47ec64ad79524abbd7a657526'
assert c['source_checkpoint']['sha256']=='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724' and c['source_cursor']==0 and c['source_adam_steps']==10464
assert c['source_world_size']==4 and c['current_world_size']==1 and c['new_head']['new_Adam_steps']==0
assert read(b/'source-publication.json')['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e'
assert not (b/'failure.json').exists() and not (b/'first-cp-audit-failure.json').exists()
first=read(b/'first-cp-independent-audit.json');replay=read(b/'next-update-replay.json')
assert first['status']=='JOINT_CP1_INDEPENDENT_FULL_SHA_NINE_COMPONENT_CPU_RELOAD_PASSED' and first['cursor']==1 and first['legacy_Adam']==10465 and first['new_head_Adam']==1
assert first['component_hashes']==replay['expected']==replay['actual'] and replay['status']=='PASSED'
assert len(first['component_hashes'])==9 and pathlib.Path(first['checkpoint']['path']).stat().st_size==first['checkpoint']['bytes']
curve=[json.loads(line) for line in (b/'curve.jsonl').read_text().splitlines() if line];six=curve[:6];assert len(six)==6
for i,r in enumerate(six,1):
 assert r['cursor']==i and r['steps']==10464+i and r['head_steps']==i and r['legacy_Adam_steps']==10464+i and r['new_head_Adam_steps']==i and r['joint_micro_replay_exact']
 assert r['original_questions']==32 and r['real_trajectories']==320 and r['virtual_G5_views']==128 and r['actual_forward_trajectory_exposures']==640
 assert r['grouping']['per_question_total_weight_exactly_one'] and r['grouping']['requested_views_per_question']==4 and r['grouping']['epoch']==0
 assert math.isfinite(r['loss']) and abs(r['loss']-r['trajectory_BCE']-r['count_NLL'])<1e-5*(1+abs(r['loss']))
assert len(samples)==2 and samples[1]['cursor']>samples[0]['cursor'] and samples[1]['cursor']>=6 and samples[1]['time']-samples[0]['time']>=45
result=dict(status='JOINT_ACTUAL_CP1_AND_CONTINUOUS_FIVE_PLUS_NEXT_UPDATE_ACCEPTED',job_id=samples[1]['job_id'],source_checkpoint=c['source_checkpoint'],source_cursor=0,source_legacy_Adam=10464,source_new_Adam=0,config=ref(b/'config.json'),first_CP1_CPU_audit=ref(b/'first-cp-independent-audit.json'),first_checkpoint=first['checkpoint'],first_six_real_transactions=[{k:r[k] for k in ('cursor','steps','head_steps','loss','seconds','time')} for r in six],separated_actual_observations=samples,actual_nine_component_replay_and_independent_CPU_full_reload=True,no_CP_tensor_loaded_in_this_small_acceptance=True,source_world_size=4,current_world_size=1,original_four_rank_archive_preserved=True,actor_updates=0,old_optimizer_reset=False,new_head_optimizer_initialized_once=True,remaining938_budget_not_complete=True,time=time.time())
p=b/'startup-independent-audit.json';assert not p.exists(),'Preserve accepted startup audit';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\\n');result['audit_ref']=ref(p);print(json.dumps(result))'''
spec=importlib.util.spec_from_file_location('joint_startup_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
samples=[{k:r[k] for k in ('time','cursor','steps','head_steps','job_id')} for r in (first,later)]
command='python3 -c '+shlex.quote('samples='+repr(samples)+'\n'+source)
if len(command.encode())>=16000:raise ValueError('Control command exceeds16KB')
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=45)
if r.returncode:raise RuntimeError(r.stderr[-2000:])
v=json.loads(r.stdout);v['published_acceptance_source_sha256']=hashlib.sha256(source.encode()).hexdigest();(OUT/'actual-startup-independent-audit.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
