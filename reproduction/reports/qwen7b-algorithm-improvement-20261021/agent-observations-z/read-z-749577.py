"""Read-only Z749577 own progress; no checkpoint contents or final data."""
import pathlib,json,subprocess,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1/z_mse_anchor');JOB='749577'
def read(p):return json.loads(p.read_bytes())
out=dict(time=time.time(),job_id=JOB,checkpoint_contents_opened=False,final4096_read=False)
for name in('controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
 p=B/name
 if p.exists():out[name]=read(p)
controller=out.get('controller.json',{});assert not controller or controller['job_id']==JOB,'A different actual Z writer must be reconciled with Root'
for key,args in(('sacct',['sacct','-X','-n','-P','-j',JOB,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList']),('queue',['squeue','-h','-j',JOB,'-o','%i|%T|%R|%S|%P|%q'])):
 z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);out[key]=z.stdout.strip()if not z.returncode else dict(error=z.stderr[-500:])
for p in sorted(B.glob('coverage-epoch-*.json')):out[p.name]=read(p)
for p in sorted(B.glob('evaluation-epoch-*.json')):
 v=read(p);out[p.name]=dict(epoch=v['epoch'],cursor=v['cursor'],seconds=v['seconds'],metrics={role:{state:{k:m[k]for k in('variance_MSE','variance_MAE','variance_Spearman','variance_Pearson','Top512','classwise','predeclared_G5_classification_diagnostic')if k in m}for state,m in states.items()}for role,states in v['metrics'].items()})
newfail=[];oldfail=[]
for p in sorted(B.glob('failure-rank-*.json')):
 v=read(p);v=dict(path=str(p),time=v.get('time'),traceback=v.get('traceback','')[-1800:])
 (newfail if v.get('time',0)>=controller.get('time',0)else oldfail).append(v)
out['new_writer_failures']=newfail;out['old_writer_failure_count']=len(oldfail)
p=B/'curve.jsonl';last=[]
if p.exists():
 with p.open()as f:
  for line in f:
   try:v=json.loads(line)
   except ValueError:continue
   last.append({k:v[k]for k in('time','epoch','cursor','training_loss','target4variance_MSE','variance_MSE','variance_MAE','alignment','actual_global_batch','native_gradient_L2','head_gradient_L2')if k in v});last=last[-8:]
out['last8_updates']=last
log=B/('train-'+JOB+'.log')
if log.exists():
 with log.open('rb')as f:f.seek(max(0,log.stat().st_size-3000));tail=f.read().decode(errors='replace')
 out['actual_log_error_tail']=tail if any(w in tail for w in('Traceback (most recent','RuntimeError:','out of memory','NCCL error','DUE TO TIME LIMIT'))else None
print(json.dumps(out))
