import pathlib,json,time,os
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/training-extraG5-matched6528-20261021/materialized')
out=dict(epoch=time.time(),complete=(b/'bank/complete.json').exists(),processes=[])
p=b/'g10-bank-merge.log';out['log_bytes']=p.stat().st_size if p.exists() else None
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-2000));last=f.read().decode(errors='replace')
 out['error_tail']=last if 'Traceback' in last or 'Error' in last else None
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  raw=(p/'cmdline').read_bytes()
  if p.stat().st_uid!=os.getuid() or b'-m\x00reproduction.operations.qwen7b_warmup.extra_g5_merge' not in raw:continue
  own=dict(pid=int(p.name),start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19]);own['io']={k:int(v) for k,v in (line.split(':',1) for line in (p/'io').read_text().splitlines())};out['processes'].append(own)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
if out['complete']:
 v=json.loads((b/'bank/complete.json').read_text());out['accepted']={k:v[k] for k in ('status','questions','trajectories')}
print(json.dumps(out))
