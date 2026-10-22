"""Read only small queue/progress/error summaries, never model or token data."""
import pathlib,json,subprocess,time,collections
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-global64-window2018-L15P80-8192-20261023-v1');out={'time':time.time(),'cells':{}};jobs=[]
def read(p):return json.loads(p.read_bytes())
def cmd(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-1000:]}
for cell in ('current64','window2018'):
 d=b/cell;c={}
 for n in ('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
  if(d/n).is_file():c[n]=read(d/n)
 if 'root-submission.json'in c:jobs.append(c['root-submission.json']['job_id'])
 c['failures']={p.name:read(p)for p in d.glob('failure*.json')}
 c['evaluation_files']=[p.name for p in d.glob('evaluation-epoch-*.json')]
 c['audit_files']=[p.name for p in d.glob('score-window-audit-epoch-*.json')]
 if(d/'curve.jsonl').is_file():
  with(d/'curve.jsonl').open()as f:last=collections.deque(f,maxlen=6)
  keep=('time','epoch','cursor','actual_Adam_updates','EMA_updates','unscaled_bound_loss','training_loss_scaled','current_logit_gradient_L2','native_gradient_L2','head_gradient_L2','N_eff','K','detached_historical_questions','Adam_and_EMA_step_performed','margin_satisfied_batches','cached_only_zero_current_gradient_batches')
  c['last_six_real_attempts']=[{k:v for k,v in json.loads(line).items()if k in keep}for line in last]
 logs=sorted(d.glob('train-*.log'))
 if logs:c['log_tail']=logs[-1].read_text(errors='replace')[-2500:]
 out['cells'][cell]=c
if jobs:
 out['accounting']=cmd(['sacct','-X','-n','-P','-j',','.join(jobs),'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])
 out['queue']=cmd(['squeue','--start','-h','-j',','.join(jobs),'-o','%i|%j|%P|%T|%S|%R'])
print(json.dumps(out))
