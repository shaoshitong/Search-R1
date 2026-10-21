"""Bounded CPU-only numerical qualification; no model/scoring/runner activity."""
import argparse,base64,gzip,hashlib,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--commit',required=True);v=p.parse_args();payload={}
for name in ('prior_logit_event_loss.py','test_prior_logit_event_loss.py'):
 path='reproduction/operations/qwen7b_warmup/'+name;data=subprocess.check_output(['git','show',v.commit+':'+path],cwd=ROOT)
 if data!=(ROOT/path).read_bytes():raise ValueError('Exact published math source required')
 payload[name]=base64.b64encode(data).decode()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=ROOT).decode().split()[0]
if remote!=v.commit:raise ValueError('Source not remotely verified')
values=dict(commit=v.commit,payload=base64.b64encode(gzip.compress(json.dumps(payload).encode(),mtime=0)).decode())
body='''import pathlib,json,gzip,base64,hashlib,fcntl,subprocess,os,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prior-logit-event-loss-math-20261021');b.mkdir(exist_ok=True);lock=(b/'qualify.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);code=b/('code-'+values['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
files=json.loads(gzip.decompress(base64.b64decode(values['payload'])))
for name,data in files.items():
 data=base64.b64decode(data);p=code/name
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data);p.chmod(0o444)
receipt=code/'qualification.json'
if receipt.exists():raise RuntimeError('Existing immutable qualification; read it before any repeat')
pub=json.loads(pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2/source-publication.json').read_text());assert pub['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e'
fixture="""import os,resource,importlib.util,sys,unittest,json,time
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(45,45))
import reproduction.operations.qwen7b_warmup as pkg
from pathlib import Path
root=Path(CODE);started=time.time()
for name in ('prior_logit_event_loss','test_prior_logit_event_loss'):
 full='reproduction.operations.qwen7b_warmup.'+name;spec=importlib.util.spec_from_file_location(full,str(root/(name+'.py')));m=importlib.util.module_from_spec(spec);sys.modules[full]=m;setattr(pkg,name,m);spec.loader.exec_module(m)
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m));assert r.wasSuccessful()
loss=sys.modules['reproduction.operations.qwen7b_warmup.prior_logit_event_loss'];torch=loss.torch
for mode in loss.MODES:
 pi=loss.frozen_full938_train_prior(mode);Ks=[0,1]if mode=='binary'else[0,1,5];labels=torch.tensor([[int(j<k)for j in range(5)]for k in Ks]);z=torch.tensor([.25,.125,.125,.125,.125,.25],dtype=torch.float64).log().requires_grad_();expected=(loss.prior_adjusted_view_losses(z.expand(len(Ks),6),labels,pi,mode)*pi).sum();g=torch.autograd.grad(expected,z)[0];torch.testing.assert_close(g,torch.zeros_like(g),atol=1e-14,rtol=0)
print(json.dumps(dict(status='PRIOR_LOGIT_COARSE_EVENT_MATH_CPU_QUALIFIED_PREPARE_ONLY',tests=r.testsRun,actual_frozen_training_prior_modes=2,actual_constant_tau_expected_gradient_zero=True,numeric_training_prior_exported=False,GPU=0,new_model_scoring=False,runner_or_screen_modified=False,final_pool_read=False,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
""".replace('CODE',repr(str(code)))
env=dict(os.environ,QWEN7B_CODE_ROOT=pub['code_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(['bash',pub['code_root']+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u','-c',fixture],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,universal_newlines=True,timeout=55);log=code/'fixtures.log';log.write_text(r.stdout)
if r.returncode:raise RuntimeError(r.stdout[-2600:])
result=json.loads([s for s in r.stdout.splitlines()if s.startswith('{')][-1]);result.update(source_commit=values['commit'],files={n:proof(code/n)for n in files},log=proof(log));receipt.write_text(json.dumps(result,sort_keys=True)+'\\n');print(json.dumps(dict(result=result,receipt=proof(receipt))))
'''
command='python3 -c '+shlex.quote('values='+repr(values)+'\n'+body)
if len(command.encode())>=16000:raise ValueError('Code-only command exceeds16KB')
spec=importlib.util.spec_from_file_location('math_only_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=70)
if r.returncode:raise RuntimeError(r.stderr[-2800:])
result=json.loads(r.stdout.splitlines()[-1]);(OUT/'actual-qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
