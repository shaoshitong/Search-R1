import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1');out=dict(epoch=time.time(),base=str(b))
for name in('submission-result.json','controller.json','progress.json','latest.json','failure.json','complete.json','startup-binding-migration.json'):
 try:
  value=json.loads((b/name).read_bytes())
  if name=='latest.json':value={k:value.get(k)for k in('cursor','native_steps','head_steps','path','sha256','bytes')}
  if name=='startup-binding-migration.json':value={k:value.get(k)for k in('status','source_cursor','raw_EMA_twoAdam_RNG_buffers_and_sampling_position_preserved','sampler_binding_changed_for_user_batch128')}
  out[name[:-5]]=value
 except FileNotFoundError:pass
if out.get('submission-result'):
 job=out['submission-result']['job_id'];r=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=8);out['sacct']=r.stdout.strip();p=b/'logs'/('train-'+job+'.log')
 if p.exists():
  with p.open('rb')as f:f.seek(0,2);n=f.tell();f.seek(max(0,n-2200));out['log_tail']=f.read().decode(errors='replace')
p=b/'curve.jsonl'
if p.exists():
 with p.open('rb')as f:f.seek(0,2);n=f.tell();f.seek(max(0,n-50000));lines=f.read().splitlines()
 rows=[]
 for line in lines:
  try:rows.append(json.loads(line))
  except ValueError:pass
 out['last_rows']=[{k:x.get(k)for k in('cursor','time','questions','micro_questions','seconds','pre_Adam_OOM_retries','prefetch')}for x in rows[-12:]]
print(json.dumps(out))
