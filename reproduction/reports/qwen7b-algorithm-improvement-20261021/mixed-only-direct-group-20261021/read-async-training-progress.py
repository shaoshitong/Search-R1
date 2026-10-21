"""Read bounded own-job status, complete accepted clocks and data-queue timings."""
import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-fast-v1')
out=dict(epoch=time.time(),base=str(b))
for name in ('controller.json','progress.json','latest.json','failure.json','startup-qualification.json','first-fast-update-replay.json','first-fast-cp-independent-audit.json','fast-durable-next-update-replay.json','startup-binding-migration.json','reader-integrity.json'):
 p=b/name
 try:
  value=json.loads(p.read_bytes())
  if name=='latest.json':value={k:value.get(k)for k in('cursor','native_steps','head_steps','path','sha256','bytes','full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed')}
  if name=='startup-binding-migration.json':value={k:value.get(k)for k in('status','source_cursor','global_Adam_sampler_RNG_preserved','source_accepted_hashes','new_accepted_hashes')}
  if name.endswith('replay.json'):value={k:value.get(k)for k in('status','cursor','source_cursor','micro_questions','native_changed','group_head_changed')}
  out[name[:-5]]=value
 except FileNotFoundError:pass
r=subprocess.run(['sacct','-X','-n','-P','-j','746220','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=8);out['sacct']=r.stdout.strip()
curve=b/'curve.jsonl'
if curve.exists():
 with curve.open('rb')as f:f.seek(0,2);end=f.tell();f.seek(max(0,end-65536));raw=f.read()
 lines=raw.splitlines();rows=[]
 for line in lines:
  try:rows.append(json.loads(line))
  except ValueError:pass
 out['last_rows']=[{k:r.get(k)for k in('cursor','seconds','time','binary_Mixed_BCE','micro_questions','native_forward_calls','prefetch')}for r in rows[-32:]]
log=b/'logs/train-746220.log'
if log.exists():
 with log.open('rb')as f:f.seek(0,2);end=f.tell();f.seek(max(0,end-1800));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
