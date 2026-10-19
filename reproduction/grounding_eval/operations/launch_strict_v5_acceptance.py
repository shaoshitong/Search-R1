"""Detached, no-GPU independent strict-v5 CPU acceptance and preserved-pilot check."""
import argparse,importlib.util,subprocess,shlex,json,pathlib
p=argparse.ArgumentParser(__doc__);p.add_argument('--method',choices=['Search-G1','Soft-to-Hard','OPSD3304'],default='Search-G1');a=p.parse_args();prefix={'Search-G1':'search-g1','Soft-to-Hard':'soft-to-hard','OPSD3304':'opsd3304'}[a.method]
code=r'''
import pathlib,json,subprocess,os,time,fcntl,sys,hashlib
prefix=PREFIX;b=pathlib.Path('/data/shaoshitong/grounding-eval-20261019');cf=b/(prefix+'-cf-strict-v5');full=b/(prefix+'-cf-strict-v5-full-control');out=b/(prefix+'-strict-v5-acceptance-control');target=cf/'independent-acceptance-20261019.json';out.mkdir(exist_ok=True)
f=open(out/'launch.lock','a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((full/'complete.json').read_text())['state']=='CF_ONLY_COMPLETE'
preserved=json.loads((full/'launch.json').read_text())['preserved_result_file_sha256']
assert all(hashlib.sha256((cf/'results'/name).read_bytes()).hexdigest()==s for name,s in preserved.items())
if target.exists():print(json.dumps({'state':'EXISTING_COMPLETE','output':str(target)}));raise SystemExit(0)
if(out/'launch.json').exists():print((out/'launch.json').read_text());raise SystemExit(0)
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
 except OSError:continue
 if 'verify_completed_strict_v5.py'in cmd and str(cf/'manifest.json')in cmd and ' -c 'not in cmd:print(json.dumps({'state':'EXISTING_FOREGROUND_INSPECT_ONLY','pid':proc.name}));raise SystemExit(0)
command=['/data/search-g1/env/bin/python','-B',str(b/'audit-runtime-14525dd6/verify_completed_strict_v5.py'),'--manifest',str(cf/'manifest.json'),'--runtime',str(b/'runtime-16c36aa7'),'--output',str(target)]
wrapper="import pathlib,json,subprocess,os,time\no=pathlib.Path("+repr(str(out))+")\ncommand="+repr(command)+"\nwith(o/'verification.log').open('ab')as f:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'));rc=p.wait()\n(o/'exit.json').write_text(json.dumps({'exit_code':rc,'completed_at':time.time(),'child_pid':p.pid,'command':command}))\n"
(out/'controller.py').write_text(wrapper)
with(out/'controller.log').open('xb')as log:p=subprocess.Popen(['/usr/bin/python3',str(out/'controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch={'pid':p.pid,'launched_at':time.time(),'command':command,'cpu_only':True,'preserved_result_hashes_checked':len(preserved),'preserved_results_unchanged':True,'verifier_sha256':hashlib.sha256((b/'audit-runtime-14525dd6/verify_completed_strict_v5.py').read_bytes()).hexdigest()}
(out/'launch.json').write_text(json.dumps(launch));print(json.dumps(launch))
'''.replace('PREFIX',repr(prefix))
compile(code,'strict-v5-cpu-acceptance-launch','exec');s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
r=subprocess.run(ctl.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=30);d={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-1000:]};pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/grounding-eval-20261019/'+prefix+'-strict-v5-acceptance-launch.json').write_text(json.dumps(d,indent=2));print(json.dumps(d));raise SystemExit(r.returncode)
