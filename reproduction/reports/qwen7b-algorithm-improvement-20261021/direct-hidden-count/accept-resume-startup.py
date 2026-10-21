"""Small independent actual continuation first-six-transaction acceptance."""
import importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
observations=[json.loads(line) for line in (OUT/'actual-resume-observer/observations.jsonl').read_text().splitlines()]
valid=[x for x in observations if x.get('handoff') and x.get('cursor') is not None and not x.get('failure')]
if not valid:raise RuntimeError('Actual accepted continuation checkpoint must exist first')
first=valid[0];source_cursor=first['source_cursor']
later=next((x for x in valid if x['cursor']>=source_cursor+6 and x['cursor']>first['cursor'] and x['time']-first['time']>=45),None)
if later is None:raise RuntimeError('Wait for six actual new transactions and two positive separated observations')
source='''import pathlib,json,hashlib,time,math
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2');attempt=b/'continuations/attempt-0001-v2'
def read(p):return json.loads(pathlib.Path(p).read_text())
def ref(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
prepared=read(attempt/'preparation.json');origin=prepared['source_cursor'];handoff=read(attempt/'handoff-complete.json');replay=read(attempt/'next-update-replay.json');cpu=read(attempt/'first-cp-independent-audit.json')
assert handoff['source_cursor']==origin and handoff['accepted_cursor']==origin+1 and handoff['actual_optimizer_updates_added']==1
assert replay['status']=='PASSED' and replay['actual']==replay['expected']==cpu['accepted_hashes'] and len(cpu['accepted_hashes'])==9
assert cpu['cursor']==origin+1 and cpu['legacy_Adam_steps']==10465+origin and cpu['new_head_Adam_steps']==origin+1
assert cpu['full_CPU_reload_passed'] and cpu['full_file_SHA256_verified'] and cpu['source_all_four_RNG_preserved']
assert prepared['current_source_world_size']==prepared['current_world_size']==1 and prepared['hardware_topology_changed'] is False
assert (attempt/'job.id').read_text().strip()==samples[1]['job_id'] and samples[1]['job_id']!=prepared['original_job_id']
assert not (attempt/'failure.json').exists() and not (b/'failure.json').exists()
curve=[json.loads(line) for line in (b/'curve.jsonl').read_text().splitlines() if line];six=curve[origin:origin+6];assert len(six)==6
assert [row['cursor'] for row in curve]==list(range(1,len(curve)+1))
for i,row in enumerate(six,origin+1):
 assert row['cursor']==i and row['steps']==10464+i and row['head_steps']==i and row['legacy_Adam_steps']==10464+i and row['new_head_Adam_steps']==i
 assert row['joint_micro_replay_exact'] and row['original_questions']==32 and row['real_trajectories']==320 and row['virtual_G5_views']==128
 assert row['grouping']['per_question_total_weight_exactly_one'] and math.isfinite(row['loss'])
assert samples[1]['cursor']>=origin+6 and samples[1]['cursor']>samples[0]['cursor'] and samples[1]['time']-samples[0]['time']>=45
result=dict(status='ACTUAL_CONTINUATION_NINE_STATE_REPLAY_FULLCP_AND_SIX_NEW_TRANSACTIONS_ACCEPTED',job_id=samples[1]['job_id'],source_cursor=origin,source_checkpoint=prepared['source_checkpoint'],first_new_checkpoint=cpu,source_current_world_size=1,current_world_size=1,hardware_topology_changed=False,original_branch_world_size=4,original_four_rank_archive_preserved=True,first_six_new_transactions=[{k:row[k] for k in ('cursor','steps','head_steps','time','loss','seconds')} for row in six],observations=samples,source_prefix_retrained=False,accepted_updates_added=6,remaining_budget_not_complete=True,actor_updates=0,optimizer_reset=False,time=time.time())
p=attempt/'startup-independent-audit.json';assert not p.exists(),'Preserve existing continuation acceptance';p.write_text(json.dumps(result,indent=2)+'\\n');result['audit_ref']=ref(p);print(json.dumps(result))'''
spec=importlib.util.spec_from_file_location('resume_startup_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
samples=[{k:x[k] for k in ('time','cursor','steps','head_steps','job_id')} for x in (first,later)]
command='python3 -c '+shlex.quote('samples='+repr(samples)+'\n'+source)
if len(command.encode())>=16000:raise ValueError('Control command exceeds16KB')
result=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=45)
if result.returncode:raise RuntimeError(str(result.stderr[-1600:]))
data=json.loads(result.stdout);(OUT/'actual-resume-startup-independent-audit.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
