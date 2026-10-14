from pathlib import Path
import sys,subprocess,shlex,json
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
def call(host,code,timeout=100):
 a=control.route(host,'python3 -B -c '+shlex.quote(code));j=a.index('mgmt-host');p=a[:j];clean=[];i=0
 while i<len(p):
  if p[i]=='-o' and p[i+1].startswith(('ProxyCommand=','ProxyJump=','BindInterface=')):i+=2;continue
  clean.append(p[i]);i+=1
 r=subprocess.run(clean+a[j:],capture_output=True,text=True,timeout=timeout);assert r.returncode==0,r.stderr[-1500:];return json.loads(r.stdout)
if __name__=='__main__':
 code="""from pathlib import Path
import json,subprocess,sys,importlib.util
r=Path('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012');o={'hostname':subprocess.run(['hostname'],text=True,capture_output=True).stdout,'mixed_root_present':r.exists(),'python':sys.executable,'torch_spec':str(importlib.util.find_spec('torch')),'models':[]}
for p in [r,r/'env',r/'checkpoints/search-g1-common8/actor/global_step_50',r/'checkpoints/search-g1-common8/actor/global_step_150',Path('/mnt/lustre/shaoshitong')]:
 o['models'].append({'path':str(p),'exists':p.exists(),'resolved':str(p.resolve()),'mount':subprocess.run(['findmnt','-T',str(p),'-n','-o','SOURCE,FSTYPE,TARGET'],text=True,capture_output=True).stdout})
o['slurm']=subprocess.run(['sinfo','-N','-h','-o','%N %P %t %G'],text=True,capture_output=True).stdout
print(json.dumps(o))"""
 for host in ['h100-8','h100-5']:
  v=call(host,code);Path(__file__).with_name(host+'-replay-preflight.json').write_text(json.dumps(v,indent=2));print(host,json.dumps(v))
