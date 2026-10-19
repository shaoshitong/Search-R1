"""SSH-disconnection-safe CPU-only independent OPSD CF acceptance."""
import importlib.util,subprocess,shlex,json,pathlib
code=r'''
import pathlib,json,subprocess,os,time,fcntl
b=pathlib.Path('/data/shaoshitong/grounding-eval-20261019');out=b/'opsd3304-cf-acceptance-control';out.mkdir(exist_ok=True)
f=open(out/'launch.lock','a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(out/'launch.json').exists():print((out/'launch.json').read_text());raise SystemExit(0)
assert json.loads((b/'opsd3304-cf-pipeline-v4-recovery1/complete.json').read_text())['exit_code']==0
assert not(b/'opsd3304-cf-full-v4/independent-acceptance-20261019.json').exists()
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
 except OSError:continue
 if 'verify_completed_cf.py'in cmd and ' -c 'not in cmd:raise RuntimeError('Existing independent verifier')
command=['/data/search-g1/env/bin/python','-B',str(b/'acceptance-runtime-30242ef2/verify_completed_cf.py'),'--manifest',str(b/'opsd3304-cf-full-v4/manifest.json'),'--runtime',str(b/'runtime-21b9ccf4'),'--output',str(b/'opsd3304-cf-full-v4/independent-acceptance-20261019.json')]
wrapper="import pathlib,json,subprocess,os,time\no=pathlib.Path("+repr(str(out))+ ")\ncommand="+repr(command)+"\nwith (o/'verification.log').open('ab')as f:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'));rc=p.wait()\n(o/'exit.json').write_text(json.dumps({'exit_code':rc,'completed_at':time.time(),'child_pid':p.pid,'command':command}))\n"
(out/'controller.py').write_text(wrapper)
with(out/'controller.log').open('xb')as log:p=subprocess.Popen(['/usr/bin/python3',str(out/'controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch={'pid':p.pid,'launched_at':time.time(),'command':command,'cpu_only':True,'reason':'Prior foreground verifier SSH broken pipe; confirmed absent before new bounded retry'}
(out/'launch.json').write_text(json.dumps(launch));print(json.dumps(launch))
'''
compile(code,'cpu-acceptance-launch','exec')
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=subprocess.run(m.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=30)
d={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-1000:]};pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/grounding-eval-20261019/opsd-cf-acceptance-background-launch.json').write_text(json.dumps(d,indent=2));print(json.dumps(d));raise SystemExit(r.returncode)
