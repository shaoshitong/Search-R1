import pathlib,json,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-pilot-CPU-acceptance-20261022-v1');c=json.loads((b/'controller.json').read_bytes());p=pathlib.Path('/proc/%d'%c['pid']);r=dict(time=time.time(),controller=c,exists=p.exists())
if p.exists():
 assert 'accept_online_mixed_pilot_checkpoints.py' in(p/'cmdline').read_bytes().decode(errors='replace');r['io']={k:int(v.strip())for k,v in(line.split(':',1)for line in(p/'io').read_text().splitlines())};r['memory']={k:v.strip()for k,v in(line.split(':',1)for line in(p/'status').read_text().splitlines())if k in('VmRSS','VmHWM','VmSize','Threads')};r['stat_fields']=(p/'stat').read_text().rsplit(')',1)[1].split()[11:15]
for n in('progress.json','failure.json','run-complete.json'):
 if(b/n).exists():r[n]=json.loads((b/n).read_bytes())
print(json.dumps(r))
