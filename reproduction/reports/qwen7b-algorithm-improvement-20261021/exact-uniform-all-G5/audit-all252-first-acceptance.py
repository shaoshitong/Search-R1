"""Read-only remote full-file proof; all checkpoint/prediction bytes stay remote."""
import importlib.util
import json
import pathlib
import shlex
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[4]
CONTROL = ROOT.parents[1]
OUT = pathlib.Path(__file__).parent
observed = json.loads((OUT/'actual-observer/latest-good.json').read_text())
assert observed['latest.json']['cursor'] == 1, 'Use the accepted CP1 receipt, never invent a receipt'
receipt = observed['latest.json']
remote = r'''
import hashlib,json,pathlib,time
b=pathlib.Path(BASE)
def read(name):return json.loads((b/name).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
replay=read('next-update-replay.json');p=pathlib.Path(RECEIPT['path'])
assert replay['status']=='PASSED' and replay['actual']==replay['expected']
assert replay['source_steps']==10464 and replay['candidate_steps']==10465
assert RECEIPT['cursor']==1 and RECEIPT['steps']==10465
assert RECEIPT['accepted_hashes']==replay['actual']
assert p.stat().st_size==RECEIPT['bytes'] and sha(p)==RECEIPT['sha256']
progress=read('progress.json');assert progress['cursor']>=2
assert progress['steps']==10464+progress['cursor']
curve=[json.loads(s) for s in (b/'curve.jsonl').read_text().splitlines() if s]
assert [r['cursor'] for r in curve]==list(range(1,len(curve)+1))
assert all(r['steps']==10464+r['cursor'] and r['actor_updates']==0 for r in curve)
assert len(curve)>=2 and curve[1]['time']>curve[0]['time']
assert not (b/'failure.json').exists() and not (b/'interrupted.json').exists()
log=b/'logs/train-743417.log';tail=log.read_text(errors='replace')[-16384:]
assert 'Traceback (most recent call last)' not in tail
assert 'RuntimeError:' not in tail
result=dict(status='ACTUAL_ALL252_CP1_FULL_SHA_DOUBLE_REPLAY_AND_FURTHER_UPDATES_ACCEPTED',
 time=time.time(),checkpoint=RECEIPT,six_component_double_replay_exact=True,
 fresh_full_checkpoint_sha256=RECEIPT['sha256'],
 production_cpu_reload_and_six_component_gate_precedes_receipt=True,
 observed_updates=len(curve),latest_observed_cursor=progress['cursor'],
 latest_Adam_steps=progress['steps'],first_update_time=curve[0]['time'],
 second_update_time=curve[1]['time'],last_update_time=progress['time'],
 seconds_min=min(r['seconds'] for r in curve),seconds_max=max(r['seconds'] for r in curve),
 CUDA_peak_allocated_bytes=max(r['CUDA_peak_allocated_bytes'] for r in curve),
 CUDA_peak_reserved_bytes=max(r['CUDA_peak_reserved_bytes'] for r in curve),
 actor_updates=0,failure_receipt_present=False,error_markers_in_current_log_tail=False,
 qualification_is_not_final_budget_or_evaluation=True)
dest=b/'first-accepted-update-audit.json'
assert not dest.exists(),'Do not overwrite a previous independent audit'
dest.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
result['audit_ref']=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest))
print(json.dumps(result,sort_keys=True))
'''
remote = ('BASE='+repr(observed['base'])+'\nRECEIPT='+repr(receipt)+'\n'+remote)
spec=importlib.util.spec_from_file_location('all252_first_acceptance_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
r=ctl.run('python3 -c '+shlex.quote(remote),account='mou176',source_address=ip,timeout=60)
if r.returncode:raise SystemExit(r.stderr[-2000:])
result=json.loads(r.stdout.splitlines()[-1])
(OUT/'actual-first-accepted-update-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,sort_keys=True))
