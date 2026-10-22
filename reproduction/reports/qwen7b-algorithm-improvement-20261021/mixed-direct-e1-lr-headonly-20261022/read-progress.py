import json,pathlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-e1-lr-headonly-20261022-v1');out={'observed_epoch':time.time(),'base':str(b)}
for name in ('submission-result.json','controller.json','progress.json','latest.json','failure.json','complete.json','source-restore-complete.json','startup-complete.json','startup-binding-migration.json','phase-transition.json'):
 p=b/name
 if p.exists():
  value=json.loads(p.read_bytes())
  if name=='latest.json':value={k:value.get(k)for k in ('path','bytes','sha256','cursor','native_steps','head_steps','phase','stage','training_phase','phase_state','native_learning_rate','head_learning_rate','native_frozen','learning_rates','raw_native_frozen') if k in value}
  out[name[:-5]]=value
job=out.get('submission-result',{}).get('job_id')or out.get('controller',{}).get('job_id')
if job:
 r=subprocess.run(['sacct','-X','-n','-P','-j',str(job),'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10);out['sacct']=r.stdout.strip();out['sacct_error']=r.stderr.strip()
 for p in sorted((b/'logs').glob('*'+str(job)+'*.log')):
  with p.open('rb')as f:f.seek(0,2);end=f.tell();f.seek(max(0,end-2500));out['log_tail']=f.read().decode(errors='replace')
p=b/'curve.jsonl'
if p.exists():
 with p.open('rb')as f:f.seek(0,2);end=f.tell();f.seek(max(0,end-50000));lines=f.read().splitlines()
 rows=[]
 for line in lines:
  try:rows.append(json.loads(line))
  except ValueError:pass
 out['last_rows']=[{k:x[k]for k in ('cursor','time','native_steps','head_steps','phase','native_lr','head_lr','seconds','loss','native_gradient_L2','group_head_gradient_L2','mixed_balanced_accuracy','micro_questions','questions')if k in x}for x in rows[-10:]]
for i in (2,3):
 p=b/'evaluations'/('epoch-%02d'%i)/'complete.json'
 if p.exists():out['epoch_%s_evaluation'%i]=json.loads(p.read_bytes())
print(json.dumps(out))
