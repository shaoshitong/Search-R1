"""Actual CPU/full-SHA readback of the NEW Prefix20 probe; reuse actorCP50 seals."""
import hashlib
import json
import shlex
import subprocess
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT

QUERY=r'''
import os,sys,json,time,hashlib,gc
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3')
if not (b/'preparation/progressive-bootstrap-complete.json').exists():
 print(json.dumps(dict(status='WAITING_FOR_REAL_THREE_EPOCH_BOOTSTRAP')));raise SystemExit(0)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
os.environ['DYNRPO_PROGRESSIVE_PREFIX']='cp50_prefix20_40_80_three_epoch_v1';os.environ['DYNRPO_PREFIX_TOKENS']='20'
sys.path.insert(0,str(b/'repo'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()
def rd(p):return json.loads(Path(p).read_text())
c=rd(b/'preparation/launch-config.json');j=rd(b/'outers/outer-000050/complete.json');s=c['resume_joint']
assert sha(s['qualification_proof'])==s['qualification_proof_sha256']
qualified=rd(s['qualification_proof'])
for row in qualified['verified_files']:
 st=Path(row['path']).stat()
 assert (st.st_ino,st.st_size,st.st_mtime_ns)==(row['inode'],row['bytes'],row['mtime_ns'])
 assert str(os.statvfs(row['path']).f_fsid)==str(row['shared_filesystem_fsid'])
probe=j['probe'];path=Path(j['probe_snapshot']);before=path.stat()
assert before.st_size==probe['bytes'] and sha(path)==probe['sha256']
proof=b/'preparation/progressive-bootstrap-probe-full-sha-cpu.json'
if proof.exists():
 old=rd(proof);assert old['probe_checkpoint_sha256']==probe['sha256']
 print(json.dumps(old));raise SystemExit(0)
import torch
torch.set_num_threads(2)
from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import probe_state
saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
summary=probe_state(saved,probe,c,50,{50:j})
after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
assert j['actor']['actual_adam_steps_per_rank']==[1000]*4 and j['actor']['scheduler_completed_outer']==50
assert j['probe_metrics']['epochs']==3 and j['probe_metrics']['new_adam_steps_per_rank']==150
assert j['probe_metrics']['actor_version']==j['actor']['actor_version']
epochs=[rd(b/'outers/outer-000050/probe-update'/('epoch-%03d.json'%n)) for n in (1,2,3)]
assert [e['actual_adam_steps_per_rank'] for e in epochs]==[[300]*4,[350]*4,[400]*4]
value=dict(status='CP50_ACTOR_EXISTING_SEALS_AND_NEW_PREFIX20_PROBE_FULL_SHA_CPU_PASS',
 source_runtime_commit=c['source_commit'],source_joint_sha256=s['source_joint_sha256'],source_full_cpu_proof=s['qualification_proof'],
 source_full_cpu_proof_sha256=s['qualification_proof_sha256'],source_actor_full_sha_repeated=False,
 source_qualified_file_stat_seals_preserved=len(qualified['verified_files']),
 probe_checkpoint_path=str(path),probe_checkpoint_sha256=probe['sha256'],probe_full_sha_recomputed=True,
 probe_actual_cpu_deserialization=True,probe_summary=summary,prefix_tokens=20,head_max_tokens=293,
 actor_actual_adam_per_rank=[1000]*4,probe_actual_adam_per_rank=[400]*4,probe_epochs=3,
 real_actor_version_matches_probe_data=True,source_actor_updates_reexecuted=0,
 epoch_boundary_actual_steps=[300,350,400],operator_source_sha256=OPERATOR_SHA,checked_unix=time.time())
with proof.open('x') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
print(json.dumps(value))
'''

def main():
    code=QUERY.replace('OPERATOR_SHA',repr(hashlib.sha256(QUERY.encode()).hexdigest()))
    address=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
    r=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=address,timeout=55)
    if r.returncode:raise RuntimeError(r.stdout.decode()[-5000:]+r.stderr.decode()[-1000:])
    value=json.loads(r.stdout)
    (OUTPUT/'progressive-cp50-bootstrap-readback.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))

if __name__=='__main__':main()
