import pathlib,json,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
specs=[('current64_seed2','749965','online-packed-current64-seed19137-plan2026102302-L15P80-8192-20261023-v1/current64',5),('MSE_M5_seed2','750001','online-packed-matched-MSE-M5-seed19137-plan2026102302-L15P80-8192-20261023-v1/mse-M5',7),('window_seed2_v1_failed','750023','online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v1/window2018',10),('full_Z_seed2','749935','online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1/z_mse_anchor',10)]
def proof(p):
 import hashlib
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
x=dict(time=time.time(),metadata_only=True,checkpoint_tensors_opened=False,final4096_read=False,experiments={})
for name,job,path,selected_epoch in specs:
 b=A/path;p=subprocess.run(['sacct','-n','-X','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList','-P'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);o=dict(job_id=job,accounting=p.stdout.strip(),accounting_error=p.stderr.strip(),fixed_replication_epoch=selected_epoch,fixed_replication_state='raw')
 for rank in(0,1):
  f=b/('progress-rank-%d.json'%rank)
  if f.exists():
   row=json.loads(f.read_bytes());o['rank%d'%rank]={key:row[key]for key in('time','epoch','cursor','actual_Adam_updates','EMA_updates','batch_attempts','margin_satisfied_batches','cached_only_zero_current_gradient_batches','N_eff','detached_historical_questions')if key in row}
 for pattern in('failure-rank-*.json',):
  for f in b.glob(pattern):o[f.name]=json.loads(f.read_bytes())
 f=b/('evaluation-epoch-%02d.json'%selected_epoch)
 if f.exists():o['selected_evaluation']=json.loads(f.read_bytes());o['selected_evaluation_proof']=proof(f)
 f=b/'complete.json'
 if f.exists():o['complete']=json.loads(f.read_bytes());o['complete_proof']=proof(f)
 x['experiments'][name]=o
print(json.dumps(x))
