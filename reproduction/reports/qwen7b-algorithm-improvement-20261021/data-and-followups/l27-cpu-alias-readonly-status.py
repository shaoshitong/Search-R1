import pathlib,json,time,os
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
b=B/'algorithm-improvement-20261021/l27-readout128-v1'
read=lambda p:json.loads(pathlib.Path(p).read_text())
select=lambda v,keys:{k:v[k] for k in keys if k in v}
out=dict(observed_epoch=time.time(),base=str(b))
for name in ('source-publication.json','cpu-alias-controller.json','native-index-audit.json','prepared-inputs.json','preparation.json'):
 p=b/name
 if p.exists():
  v=read(p)
  out[name]=select(v,['status','pid','proc_start_ticks','returncode','started','finished','checkpoint_tensors_loaded','commit','code_root','manifests','checked_rows','fit_questions','updates','block_expected_steps','new_head_expected_steps','checkpoint_tensors_loaded_on_login','model_tensors_loaded_on_login'])
  if 'entries' in v:out[name]['entries_count']=len(v['entries'])
  if 'variants' in v:out[name]['variants']=list(v['variants'])
  if 'aliases' in v:out[name]['aliases']=list(v['aliases'])
for name in ('last-prefix/last-scope-alias-check.json','last-prefix/failure.json','failure.json'):
 p=b/name
 if p.exists():
  v=read(p);out[name]=v if p.name!='failure.json' else select(v,['type','message','epoch_time','traceback'])
p=b/'cpu-alias-audit.log'
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2400));tail=f.read().decode(errors='replace')
 out['cpu-alias-audit.log']=dict(bytes=p.stat().st_size,mtime=p.stat().st_mtime,tail=tail)
p=b/'cpu-alias-controller.json';controller=read(p) if p.exists() else {}
pids=[controller['pid']] if 'pid' in controller else []
out['processes']={}
for pid in pids:
 root=pathlib.Path('/proc/%d'%pid);record=dict(exists=root.exists())
 if root.exists():
  stat=root.joinpath('stat').read_text().rsplit(')',1)[1].split();record['start_ticks']=stat[19]
  args=root.joinpath('cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
  record['is_expected_alias_checker']='l27_readout_train' in args and '--alias-check-only' in args
  record['controller_ticks_match']=controller.get('proc_start_ticks') is not None and str(controller['proc_start_ticks'])==stat[19]
  status={}
  for line in root.joinpath('status').read_text().splitlines():
   key,sep,value=line.partition(':')
   if key in ('Name','State','PPid','VmRSS','VmHWM','VmSize','Threads'):status[key]=value.strip()
  record['status']=status
  record['cgroup']=root.joinpath('cgroup').read_text().strip()
 out['processes'][str(pid)]=record
# Current control belongs to the same account; historical memory counters are
# reported explicitly without attributing them to this ended child.
entries=pathlib.Path('/proc/self/cgroup').read_text().splitlines();memory={}
for row in entries:
 hierarchy,controllers,path=row.split(':',2)
 if not controllers:
  base=pathlib.Path('/sys/fs/cgroup')/path.lstrip('/')
  for name in ('memory.current','memory.max','memory.peak','memory.events'):
   p=base/name
   if p.exists():memory[name]=p.read_text().strip()
 elif 'memory' in controllers.split(','):
  base=pathlib.Path('/sys/fs/cgroup/memory')/path.lstrip('/')
  for name in ('memory.usage_in_bytes','memory.limit_in_bytes','memory.max_usage_in_bytes','memory.failcnt','memory.oom_control'):
   p=base/name
   if p.exists():memory[name]=p.read_text().strip()
out['current_account_cgroup_memory_snapshot_historical_counts_not_run_attributed']=memory
print(json.dumps(out))
