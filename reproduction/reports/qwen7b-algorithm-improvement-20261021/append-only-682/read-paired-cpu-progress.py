"""Read exact finite paired wrapper/child CPU ticks; never infer remote absence."""
import argparse,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--cursor',type=int,choices=(512,768,938),required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('paired_cpu_progress_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
source='''import pathlib,json,time,os,socket
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-expanded-append-stability-1a0f5b4');work=base/('cursor-%06d-v2'%CURSOR);v=json.loads((work/'controller.json').read_text());pid=v['pid'];out=[]
pids=[pid]
try:pids+=[int(x) for x in (pathlib.Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()]
except OSError:pass
for p in pids:
 try:
  root=pathlib.Path('/proc')/str(p);cmd=(root/'cmdline').read_bytes()
  if str(base/'paired_question_stability.py').encode() not in cmd:continue
  raw=(root/'stat').read_text();fields=raw[raw.rfind(')')+2:].split();out.append(dict(pid=p,user_ticks=int(fields[11]),system_ticks=int(fields[12]),state=fields[0]))
 except OSError:pass
print(json.dumps(dict(cursor=CURSOR,host=socket.gethostname(),wrapper_pid=pid,CPU_processes_verified_by_exact_script=out,ticks_per_second=os.sysconf('SC_CLK_TCK'),result_exists=(work/'result.json').exists(),exit=json.loads((work/'worker-exit.json').read_text()) if (work/'worker-exit.json').exists() else None,absence_is_not_failure_proof=True,time=time.time())))'''
r=ctl.run('python3 -c '+shlex.quote('CURSOR='+str(a.cursor)+'\n'+source),account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=30)
if r.returncode:raise RuntimeError(r.stderr[-1800:])
v=json.loads(r.stdout)
with (OUT/('actual-paired-%d-CPU-progress.jsonl'%a.cursor)).open('a') as f:f.write(json.dumps(v,sort_keys=True)+'\n')
print(json.dumps(v,sort_keys=True))
