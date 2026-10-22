import pathlib,json,subprocess,time,fcntl,shutil
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');out={}
for layer in(18,15):
 b=S/('layer%d-prefix80'%layer);job=json.loads((b/'root-submission.json').read_bytes())['job_id'];cp=json.loads((b/'latest.json').read_bytes());ps=[json.loads((b/('progress-rank-%d.json'%k)).read_bytes())for k in(0,1)];assert all(p['cursor']==128 for p in ps)and cp['path'].endswith('epoch-02.pt')and not(b/'evaluation-epoch-02.json').exists()
 hp=b/'walltime-boundary-held.json';old=json.loads(hp.read_bytes());p=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);state=p.stdout.strip()
 if layer==18:
  assert any(x.split('|')[0]==job and x.split('|')[1].startswith(('TIMEOUT','CANCELLED'))for x in state.splitlines()),state
  locks=[]
  for k in(0,1):
   f=(b/('writer-rank-%d.lock'%k)).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
  for f in locks:f.close()
  shutil.copy2(str(hp),str(b/'walltime-boundary-held-epoch1-historical.json'));h={'epoch':2,'cursor':128,'checkpoint':cp,'job_id':job,'no_live_old_writer':True,'accounting':state,'evaluation_pending':True,'time':time.time(),'discarded_optimizer_updates':0}
 else:
  assert old['epoch']==2 and old['actual_GPU_step_stopped']and all(p['freezer']=='FROZEN'for p in old['actual_GPU_rank_processes']);h=old
 h['status']='COMPLETE_EPOCH_BOUNDARY_HELD_FOR_CPU_ACCEPTANCE';hp.write_text(json.dumps(h,sort_keys=True)+'\n');out[str(layer)]={'accounting':state,'held_epoch':2,'cursor':128,'checkpoint':cp}
print(json.dumps(out))
