"""Published-code-only immutable stage and bounded no-fit CPU grid execution."""
import base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
C=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5');R=C/'work/qwen7b-probe-warmup-20261019'
TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/E2-E4-fixedG5-ensemble-validation-20261021'
COMMIT='75a9c4e9';PREREG='342cb4d6';FILES={'e2_e4_fixed_ensemble.py':('75a9c4e9','reproduction/operations/qwen7b_warmup/e2_e4_fixed_ensemble.py'),'prereg.json':('342cb4d6','reproduction/reports/qwen7b-algorithm-improvement-20261021/E2-E4-CPU-ENSEMBLE-PREREG.json')}
COMMON='''import pathlib,json,hashlib,base64,fcntl,os,subprocess,time
b=pathlib.Path(inputs['base']);b.mkdir(parents=True,exist_ok=True);root=b/('code-'+inputs['commit'][:12]);root.mkdir(exist_ok=True)
def atomic(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\\n');t.replace(p)
def read(p):return json.loads(p.read_text())
f=(b/'operator.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
'''
spec=importlib.util.spec_from_file_location('fixed_cpu_control',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);addr=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
commit=subprocess.check_output(['git','rev-parse',COMMIT],cwd=R,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R,text=True).split()[0];subprocess.run(['git','merge-base','--is-ancestor',commit,remote],cwd=R,check=True)
def call(body,extra=None):
 d=dict(base=TARGET,commit=commit);d.update(extra or {});command='python3 -c '+shlex.quote('inputs='+repr(d)+'\n'+COMMON+'\n'+body);assert len(command.encode())<16000
 result=ctl.run(command,account='mou176',source_address=addr,timeout=55)
 if result.returncode:raise RuntimeError(result.stdout.decode()[-2000:]+result.stderr.decode()[-1200:])
 return json.loads(result.stdout.decode().splitlines()[-1])
proofs={}
for name,(revision,path) in FILES.items():
 data=subprocess.check_output(['git','show',revision+':'+path],cwd=R);proofs[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),commit=subprocess.check_output(['git','rev-parse',revision],cwd=R,text=True).strip())
 for i,start in enumerate(range(0,len(data),4096)):
  chunk=data[start:start+4096];call("p=root/(inputs['name']+'.part-%04d'%inputs['index']);raw=base64.b64decode(inputs['chunk']);assert hashlib.sha256(raw).hexdigest()==inputs['sha256']\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps({'accepted_chunk':inputs['index']}))",dict(name=name,index=i,chunk=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest()))
 call("data=b''.join((root/(inputs['name']+'.part-%04d'%i)).read_bytes() for i in range(inputs['chunks']));assert len(data)==inputs['bytes'] and hashlib.sha256(data).hexdigest()==inputs['sha256'];p=root/inputs['name']\nif p.exists():assert p.read_bytes()==data\nelse:p.write_bytes(data);p.chmod(0o444)\nprint(json.dumps({'immutable_file':str(p),'sha256':inputs['sha256']}))",dict(name=name,chunks=(len(data)+4095)//4096,bytes=len(data),sha256=proofs[name]['sha256']))
body='''
source=dict(commit=inputs['commit'],remote_sha=inputs['remote_sha'],files=inputs['files'],code_root=str(root),prereg_before_combinations=True);p=b/'source-publication.json'
if p.exists():assert read(p)==source
else:atomic(p,source)
if (b/'complete.json').exists():print(json.dumps(dict(status='EXISTING_GRID_RESULT_PRESERVED',complete=read(b/'complete.json'))))
else:
 assert not (b/'cpu-intent.json').exists(),'Unknown prior execution; inspect before repeating';atomic(b/'cpu-intent.json',dict(epoch=time.time(),source_publication=source,action='EXACT_PREREGISTERED_SIX_CPU_RECIPES_ONLY'))
 with (b/'cpu.log').open('ab') as log:r=subprocess.run(['python3','-u',str(root/'e2_e4_fixed_ensemble.py'),'--prereg',str(root/'prereg.json'),'--prereg-sha256',inputs['files']['prereg.json']['sha256']],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),timeout=40)
 atomic(b/'cpu-exit.json',dict(returncode=r.returncode,epoch=time.time()));assert r.returncode==0,(b/'cpu.log').read_text()[-3000:]
 print(json.dumps(dict(status='REAL_SIX_CPU_GRID_COMPLETE',complete=read(b/'complete.json'),exit=read(b/'cpu-exit.json'))))
'''
v=call(body,dict(remote_sha=remote,files=proofs));p=C/'outputs/qwen7b-warmup-20261019/e2-e4-fixed-cpu-actual.json';p.write_text(json.dumps(v,indent=2)+'\n');(R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/actual-launches'/p.name).write_text(p.read_text())
print(json.dumps({'status':v['status'],'exit':v.get('exit'),'baselines':{n:dict(BA=x['pool']['mixed_balanced_accuracy'],Top512Mixed=x['selected512']['mixed_questions']) for n,x in v['complete']['baselines'].items()},'results':[dict(id=x['recipe']['id'],BA=x['pool']['mixed_balanced_accuracy'],AUC=x['pool']['mixed_roc_auc'],Top512Mixed=x['selected512']['mixed_questions'],joint=x['joint_strict_pass']) for x in v['complete']['results']],'joint_pass_items':v['complete']['joint_pass_items'],'winner':v['complete']['winner_validation_candidate']}))
