import pathlib,json,os,time,resource
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed-ingest-audited243-after466-20261021-45227948');v=json.loads((b/'cpu-ingest-launch.json').read_text());p=pathlib.Path('/proc/%d'%v['pid']);s=(p/'stat').read_text().rsplit(')',1)[1].split();status={k:x.strip()for k,x in(s.split(':',1)for s in(p/'status').read_text().splitlines()if':'in s)};io={k:int(x)for k,x in(s.split(':',1)for s in(p/'io').read_text().splitlines())};groups=(p/'cgroup').read_text().splitlines();cg=[]
for row in groups:
 ident,controllers,path=row.split(':',2)
 root=pathlib.Path('/sys/fs/cgroup')/('memory'if'memory'in controllers.split(',')else'')/path.lstrip('/')
 entry=dict(controllers=controllers,path=path,files={})
 for key in('memory.max','memory.current','memory.peak','memory.limit_in_bytes','memory.usage_in_bytes','memory.max_usage_in_bytes'):
  q=root/key
  if q.exists():entry['files'][key]=q.read_text().strip()
 if entry['files']:cg.append(entry)
print(json.dumps(dict(status='ACTUAL_OWN_INGEST_RESOURCE_OBSERVATION',epoch=time.time(),pid=v['pid'],same_process_birth=s[19]==v['start_ticks'],cpu_seconds=(int(s[11])+int(s[12]))/os.sysconf('SC_CLK_TCK'),VmRSS=status.get('VmRSS'),VmHWM=status.get('VmHWM'),cpu_allowed=status.get('Cpus_allowed_list'),io=io,cgroup_memory=cg,progress=json.loads((b/'progress.json').read_text()),expected_original=155050,expected_newseed=155050,expected_G10=310100)))
