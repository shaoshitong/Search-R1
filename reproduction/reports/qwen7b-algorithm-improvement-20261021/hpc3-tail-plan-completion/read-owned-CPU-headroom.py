"""Read own main process CPU/RSS only; no model or process mutation."""
import pathlib,json,importlib.util
R=pathlib.Path(__file__).resolve().parents[4]
s=importlib.util.spec_from_file_location('t',R/'reproduction/operations/qwen7b_warmup/deploy_hpc3_tail_g5.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
step='''import pathlib,time,os,json
p=pathlib.Path('/proc/300719');assert b'joint_balanced_native_main'in(p/'cmdline').read_bytes()
def snap():
 v=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(birth=v[19],ticks=int(v[11])+int(v[12]),epoch=time.time())
a=snap();time.sleep(2);b=snap();assert a['birth']==b['birth'];status={x.split(':',1)[0]:x.split(':',1)[1].strip()for x in(p/'status').read_text().splitlines()if':'in x};print(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],step=os.environ['SLURM_STEP_ID'],main_pid=300719,main_birth=a['birth'],actual_window_seconds=b['epoch']-a['epoch'],average_main_CPU_cores=(b['ticks']-a['ticks'])/os.sysconf('SC_CLK_TCK')/(b['epoch']-a['epoch']),main_RSS=status.get('VmRSS'),CPU_only=True,epoch=time.time())))
'''
code='import subprocess,json\nstep='+repr(step)+'\n'+'''r=subprocess.run(['srun','--jobid=745501','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=gpu:0','--mem=128M','env','CUDA_VISIBLE_DEVICES=','python3','-c',step],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35);assert r.returncode==0,r.stderr;print(r.stdout.strip())'''
v=t.remote(code,timeout=50);p=pathlib.Path(__file__).parent/'final54-owned-CPU-actual-headroom.json';p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
