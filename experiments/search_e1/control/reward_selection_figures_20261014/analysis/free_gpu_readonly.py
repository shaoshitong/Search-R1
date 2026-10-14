from pathlib import Path
import subprocess,shlex,sys,json,concurrent.futures
D=Path(__file__).resolve().parent;sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
code="import subprocess,json;print(json.dumps({'gpu':subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout,'apps':subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout}))"
def read(host):
 a=control.route(host,'python3 -c '+shlex.quote(code));cut=a.index('mgmt-host');prefix=a[:cut];out=[];i=0
 while i<len(prefix):
  if prefix[i]=='-o' and i+1<len(prefix) and prefix[i+1].startswith(('ProxyCommand=','ProxyJump=','BindInterface=')):i+=2;continue
  out.append(prefix[i]);i+=1
 a=out+a[cut:];p=subprocess.run(a,capture_output=True,text=True,timeout=50);return {'host':host,'rc':p.returncode,'data':json.loads(p.stdout) if p.returncode==0 else None,'error':p.stderr[-250:]}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(read,['h100-5','h100-10','h100-11']))
(D/'free-gpu-readonly.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
