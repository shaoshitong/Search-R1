import sys,json,shlex,time
from pathlib import Path
w=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6');sys.path.insert(0,str(w/'work/easy7b-fa3-fastpath-20261022'))
from reproduction.operations.easy7b_20261021 import watch_cp34_recovery_local as wc
b=wc.shared.load_bridge(w)
inner=r'''import json,os,time,socket
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r2-20261023')
r={'epoch':time.time(),'hostname':socket.gethostname(),'uid':os.getuid(),'main_pid':2936710,'processes':[],'files':{}}
assert os.getuid()==14090
rows={}
for p in Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:
  if p.stat().st_uid!=14090:continue
  raw=(p/'stat').read_text();parts=raw[raw.rfind(')')+2:].split();rows[int(p.name)]=(int(parts[1]),p,raw)
 except (OSError,ValueError):pass
selected={2936710}
for _ in range(10):
 more={pid for pid,(ppid,p,raw) in rows.items() if ppid in selected};before=len(selected);selected|=more
 if len(selected)==before:break
for pid in sorted(selected):
 if pid not in rows:continue
 ppid,p,raw=rows[pid];entry={'pid':pid,'ppid':ppid,'stat':raw}
 for name in ['wchan','cmdline','io']:
  try:entry[name]=(p/name).read_bytes().decode(errors='replace').replace('\x00',' ')
  except OSError as e:entry[name+'_error']=str(e)
 r['processes'].append(entry)
for p in (base/'logs').glob('*'):
 if p.name.startswith(('native-','pipeline','training','search-g1-common8','trainer')):
  s=p.stat();r['files'][str(p)]={'bytes':s.st_size,'mtime':s.st_mtime}
print(json.dumps(r),flush=True)
'''
remote="import subprocess,json\nr=subprocess.run("+repr(['srun','--jobid=749598','--overlap','--exact','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=none','env','CUDA_VISIBLE_DEVICES=','/usr/bin/python3','-B','-c',inner])+",capture_output=True,text=True,timeout=30)\nprint(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))"
r=b.run('jxu957',shlex.join(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',remote]),'',str(w/'work'),login_host='10.120.48.31',expected_host_identity={'hostname':'ACD-Login-1','boot_id':'a27512a3-24a0-4ca3-aa92-98d2a5256d9a','pid_namespace':'pid:[4026531836]'},observation_timeout=50)
p=w/'outputs/easy2560-qwen7b-20261021'/('fastpath-native-prefix-r2-entry-proc-'+str(time.time_ns())+'.json');p.write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');print(p);print(r.stdout);print(r.stderr[-1000:]);sys.exit(r.returncode)
