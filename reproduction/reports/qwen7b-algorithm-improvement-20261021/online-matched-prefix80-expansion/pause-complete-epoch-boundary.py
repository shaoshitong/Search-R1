"""Root-only short hold of owned ranks after a published complete epoch.

No partial epoch rollback. Resume or release only after separate CPU acceptance.
The active goal was created at1792675859; this is repair of insufficient job
walltime, not a human pause/cancellation of that goal or any foreign job.
"""
import pathlib,json,subprocess,time,fcntl
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');out={'time':time.time(),'jobs':{}}
guard=(S/'prefix80-submit.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(p.read_bytes())
def run(cmd):
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert p.returncode==0,p.stdout+p.stderr;return p.stdout.strip()
for layer in (18,15):
 b=S/('layer%d-prefix80'%layer);receipt=read(b/'root-submission.json');job=receipt['job_id'];record=b/'walltime-boundary-held.json'
 if record.exists():out['jobs'][str(layer)]=read(record);continue
 if not(b/'latest.json').exists():out['jobs'][str(layer)]={'job':job,'status':'WAIT_COMPLETE_CHECKPOINT'};continue
 cp=read(b/'latest.json');path=pathlib.Path(cp['path']);epoch=int(path.stem.rsplit('-',1)[1]);cursor=64*epoch;ev=b/('evaluation-epoch-%02d.json'%epoch)
 if epoch not in(1,2)or path!=b/('checkpoint-epoch-%02d.pt'%epoch):out['jobs'][str(layer)]={'job':job,'status':'NO_EARLY_HANDOFF_NEEDED'};continue
 if ev.exists()or any(read(b/('progress-rank-%d.json'%rank))['cursor']!=cursor for rank in(0,1)):
  out['jobs'][str(layer)]={'job':job,'status':'TRAINING_OR_COMPLETED_EVAL_WAIT_NEXT_BOUNDARY','published_epoch':epoch};continue
 controller=read(b/'controller.json');assert str(controller['job_id'])==job and controller['layer']==layer and controller['prefix']==80
 actual=run(['squeue','-h','-j',job,'-o','%T|%j']);assert actual=='RUNNING|'+receipt['name'],actual
 hold={'status':'OWN_COMPLETE_EPOCH_STOP_REQUESTED','job_id':job,'layer':layer,'prefix':80,'epoch':epoch,'cursor':cursor,'checkpoint':cp,'time':time.time(),'reason':'Running walltime extension denied; preserve every completed update before90min continuation','human_goal_created_at':1792675859,'human_goal_paused':False}
 (b/'walltime-boundary-intent.json').write_text(json.dumps(hold)+'\n')
 run(['scancel','--signal=STOP','--full',job])
 try:
  assert not ev.exists()and all(read(b/('progress-rank-%d.json'%rank))['cursor']==cursor for rank in(0,1)),'Boundary passed before stop; continue original writer'
  assert read(b/'latest.json')==cp
  hold.update(status='COMPLETE_EPOCH_BOUNDARY_HELD_FOR_CPU_ACCEPTANCE',time=time.time(),evaluation_pending=True,no_partial_epoch_discarded=True)
  record.write_text(json.dumps(hold,sort_keys=True)+'\n');out['jobs'][str(layer)]=hold
 except BaseException:
  run(['scancel','--signal=CONT','--full',job]);raise
print(json.dumps(out))
