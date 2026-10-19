import sys,json,pathlib,subprocess,shlex,time
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
code=r'''import pathlib,json,time,subprocess,os
root=pathlib.Path('/mnt/lustre/shaoshitong/grounding-eval-20261019/easy-hf-upload');launch=json.loads((root/'launch.json').read_text());x={'at':time.time(),'launch':launch,'process':subprocess.run(['ps','-p',str(launch['pid']),'-o','pid,ppid,etimes,args'],capture_output=True,text=True,timeout=5).stdout}
if(root/'status.json').exists():x['status']=json.loads((root/'status.json').read_text())
else:x['status']={'state':'SOURCE_FRESH_HASH_OR_INITIALIZATION_NOT_YET_PUBLISHED'}
s=x['status'];x['log_bytes']=(root/'worker.log').stat().st_size
if s['state'].startswith('FAILED')or(len(x['process'].splitlines())<2 and s['state']!='HF_FILES_VERIFIED'):
 # Script deliberately suppresses third-party exception bodies/tracebacks.
 x['sanitized_log_tail']=(root/'worker.log').read_text()[-1200:]
print(json.dumps(x))'''
a=control.route('mgmt-host','python3 -B -c '+shlex.quote(code));a[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6'];r=subprocess.run(a,capture_output=True,text=True,timeout=20);assert r.returncode==0,'Upload observation transport failed';x=json.loads(r.stdout);D=pathlib.Path('outputs/grounding-eval-20261019');p=D/('easy-hf-upload-observation-'+str(time.time_ns())+'.json');p.write_text(json.dumps(x,indent=2)+'\n');(D/'easy-hf-upload-status-latest.json').write_text(json.dumps(x,indent=2)+'\n');s=x['status'];print(json.dumps({'at':x['at'],'state':s['state'],'pid':x['launch']['pid'],'verified_bytes':s.get('verified_bytes',0),'file_progress':s.get('file_progress',{}),'pending_operations':s.get('pending_operations'),'process':x['process'],'sanitized_log_tail':x.get('sanitized_log_tail')}))
