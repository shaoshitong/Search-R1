from pathlib import Path
from collections import deque
import json,time,subprocess
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');b=A/'online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v2/window2018';out=dict(time=time.time(),Slurm_mutated=False,model_or_checkpoint_read=False)
for n in('root-submission.json','controller.json','complete.json'):
 if(b/n).exists():out[n]=json.loads((b/n).read_bytes())
keys=('epoch','cursor','batch_attempts','actual_Adam_updates','EMA_updates','margin_satisfied_batches','cached_only_zero_current_gradient_batches','Adam_and_EMA_step_performed','rank','native_gradient_L2','head_gradient_L2','current_logit_gradient_L2','N_eff','K','current_global_questions','detached_historical_questions','prefill_rows_equal_groups','unscaled_bound_loss','training_loss_scaled','time')
for rank in(0,1):
 p=b/('progress-rank-%d.json'%rank)
 if p.exists():v=json.loads(p.read_bytes());out[p.name]={k:v[k]for k in keys if k in v}
p=b/'curve.jsonl'
if p.exists():
 with p.open()as f:tail=list(deque(f,maxlen=12))
 out['curve_tail']=[{k:v[k]for k in keys if k in v}for v in map(json.loads,tail)]
job=str(out['root-submission.json']['job_id']);assert job=='750069';out['queue']=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%T|%N|%M|%R'],universal_newlines=True,timeout=15);out['accounting']=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=15)
out['failures']={p.name:json.loads(p.read_bytes())for p in b.glob('failure-rank-*.json')};print(json.dumps(out))
