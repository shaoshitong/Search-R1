"""Authorized read-only network probes; existing allocation, zero extra GPU."""
import importlib.util,json,pathlib,shlex,subprocess
CONTROL=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
p=CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py';s=importlib.util.spec_from_file_location('ctl',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
probe=r'''
import concurrent.futures,json,os,socket,subprocess,time

def one(task):
 host,kind=task;t=time.time();v=dict(target=host,kind=kind)
 try:
  if kind=='ping1':
   r=subprocess.run(['ping','-c','1','-W','1',host],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=2);v['returncode']=r.returncode
  else:s=socket.create_connection((host,int(kind)),timeout=2);s.close();v['connected']=True
 except Exception as e:v.update(connected=False,error_type=type(e).__name__)
 v['seconds']=time.time()-t;return v
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:checks=list(pool.map(one,[(h,k) for h in ('10.120.17.102','10.120.17.122') for k in ('22','17895','ping1')]))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
for k in list(env):
 if k.lower() in ('http_proxy','https_proxy','all_proxy','no_proxy'):env.pop(k)
t=time.time();r=subprocess.run(['curl','--silent','--proxy','','--noproxy','*','--location','--max-redirs','3','--connect-timeout','3','--max-time','3','-o','/dev/null','-w','%{http_code} %{size_download}','https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/a09a35458c702b33eeacc393d103063234e8bc28/config.json'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env,timeout=5)
parts=r.stdout.decode().split();hf=dict(returncode=r.returncode,http_status=int(parts[0]) if parts else None,bytes_read=int(float(parts[1])) if len(parts)>1 else 0,seconds=time.time()-t,token_used=False,proxy_used=False)
print(json.dumps(dict(source_hostname=socket.gethostname(),epoch=time.time(),checks=checks,public_HF=hf,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),actor_updates=0,probe_updates=0)))
'''
remote='''import concurrent.futures,json,os,shlex,socket,subprocess,time
probe=%r
commands={'existing_BatchMode_Login1':['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=4','ACD-Login-1','python3 -c '+shlex.quote(probe)],'existing_743214_CPU_overlap':['srun','--jobid=743214','--overlap','--gres=gpu:0','--ntasks=1','--nodes=1','--cpus-per-task=1','--mem=256M','--time=00:00:20','--exact','python3','-c',probe]}
def run(item):
 name,args=item;t=time.time();env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
 try:
  r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=25);rows=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
  return dict(route=name,returncode=r.returncode,seconds=time.time()-t,observations=rows,error_tail=r.stderr.decode(errors='replace')[-500:] if r.returncode else None)
 except Exception as e:return dict(route=name,error_type=type(e).__name__,seconds=time.time()-t)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(run,commands.items()))
print(json.dumps(dict(control_hostname=socket.gethostname(),routes=out,extra_GPU_requested=0,training_code_or_state_modified=False,epoch=time.time())))
'''%probe
r=m.run('python3 -c '+shlex.quote(remote),account='mou176',source_address=ip,timeout=32)
if r.returncode:raise RuntimeError('Bounded network probe control failed')
v=[json.loads(x) for x in r.stdout.decode().splitlines() if x.startswith('{')][-1];out=pathlib.Path(__file__).with_name('peer-newseed-existing-allocation-and-login1-network.json');out.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
