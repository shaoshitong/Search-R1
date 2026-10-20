"""Independent small CPU acceptance of actual CP257 and six new transactions."""
import hashlib,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
observed=[json.loads(s) for s in (OUT/'actual-observer/observations.jsonl').read_text().splitlines()]
valid=[r for r in observed if r.get('cursor') is not None and not r.get('failure') and not r.get('CPU_audit_failure')]
first=valid[0];later=next((r for r in valid if r['cursor']>=262 and r['cursor']>first['cursor'] and r['time']-first['time']>=45),None)
if later is None:raise RuntimeError('Two separated real positive observations and six new updates required')
source='''import pathlib,json,hashlib,math,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-expanded-append-only-train-v2')
def read(p):return json.loads(pathlib.Path(p).read_text())
def ref(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
c=read(b/'config.json');assert ref(b/'config.json')['sha256']=='d56668bdd2e9839abe042c0d4f720225cb0cafd62f22924867dbfda883b17a80'
assert c['source_checkpoint']['sha256']=='36d9bcc3be71ba94eaca3a1e1b292c0ed9a65297fb96a746ea5295432f09c000' and c['source_cursor']==256 and c['source_adam_steps']==10720
assert c['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.) and c['source_world_size']==c['current_world_size']==1
assert read(b/'source-publication.json')['commit']=='b8c3f8a76fa195efc7c2e95cfa1a7d36669576ec'
assert not (b/'failure.json').exists() and not (b/'first-cp-audit-failure.json').exists()
first=read(b/'first-cp-independent-audit.json');replay=read(b/'next-update-replay.json')
assert first['status']=='APPEND_CP257_INDEPENDENT_FULL_SHA_AND_SIX_COMPONENT_CPU_RELOAD_PASSED' and first['cursor']==257 and first['steps']==10721
assert first['component_hashes']==replay['expected']==replay['actual'] and replay['status']=='PASSED'
assert len(first['component_hashes'])==6 and pathlib.Path(first['checkpoint']['path']).stat().st_size==first['checkpoint']['bytes']
curve=[json.loads(line) for line in (b/'curve.jsonl').read_text().splitlines() if line];six=curve[:6];assert len(six)==6
for i,r in enumerate(six,257):
 assert r['cursor']==i and r['steps']==10464+i and r['additional_completed']==i-256 and r['recomputed_logits_bit_identical']
 assert r['original_questions']==32 and r['unique_real_trajectories']==320 and r['virtual_views']==128 and r['actual_forward_trajectory_exposures']==640
 assert r['grouping']['per_question_total_weight_exactly_one'] and r['grouping']['requested_views_per_question']==4 and r['grouping']['epoch']==0
 assert math.isfinite(r['loss']) and abs(r['loss']-r['components']['trajectory']-r['components']['three_event'])<1e-5*(1+abs(r['loss']))
assert len(samples)==2 and samples[1]['cursor']>samples[0]['cursor'] and samples[1]['cursor']>=262 and samples[1]['time']-samples[0]['time']>=45
result=dict(status='APPEND_ACTUAL_CP257_AND_CONTINUOUS_FIVE_PLUS_NEXT_UPDATE_ACCEPTED',job_id='743517',source_checkpoint=c['source_checkpoint'],source_cursor=256,source_Adam=10720,config=ref(b/'config.json'),first_CP257_CPU_audit=ref(b/'first-cp-independent-audit.json'),first_checkpoint=first['checkpoint'],first_six_real_transactions=[{k:r[k] for k in ('cursor','additional_completed','steps','loss','seconds','time')} for r in six],separated_actual_observations=samples,first_next_update_six_component_replay_and_independent_CPU_reload=True,no_CP_tensor_loaded_in_this_small_acceptance=True,source_world_size=1,current_world_size=1,original_E1_archive_world_size=4,old256_retrained=False,actor_updates=0,optimizer_reset=False,remaining682_budget_not_complete=True,time=time.time())
p=b/'startup-independent-audit.json';assert not p.exists(),'Preserve accepted startup audit';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\\n');result['audit_ref']=ref(p);print(json.dumps(result))'''
spec=importlib.util.spec_from_file_location('append_startup_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
samples=[{k:r[k] for k in ('time','cursor','steps','job_id')} for r in (first,later)]
command='python3 -c '+shlex.quote('samples='+repr(samples)+'\n'+source)
if len(command.encode())>=16000:raise ValueError('Control command exceeds16KB')
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=45)
if r.returncode:raise RuntimeError(r.stderr[-2000:])
v=json.loads(r.stdout);v['published_acceptance_source_sha256']=hashlib.sha256(source.encode()).hexdigest();(OUT/'actual-startup-independent-audit.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
