"""Finite published CPU diagnostic; only public aggregates return locally."""
import argparse,base64,gzip,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
P='reproduction/operations/qwen7b_warmup/'
FILES=[P+n for n in ['direct_count_prior_decision.py','test_direct_count_prior_decision.py','direct_count_label_prior.py','e2_e4_fixed_ensemble.py','deploy_count_prior_decision.py']]
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full938-train-prior-decision-diagnostic-v1'
TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2'
BRANCH='codex/full938-prior-decision-20261021'
COMMON=r'''
import pathlib,json,hashlib,os,fcntl,time,subprocess,base64,gzip
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def atomic(p,val):
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(val,sort_keys=True)+'\n');q.replace(p)
def guard():
 f=(b/'operator.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
'''
CHUNK=r'''
lock=guard();data=base64.b64decode(v['part']);assert hashlib.sha256(data).hexdigest()==v['part_sha'];p=code/('part-%04d'%v['index'])
if p.exists():assert p.read_bytes()==data
else:p.write_bytes(data)
print(json.dumps(dict(status='OWN_CPU_DIAGNOSTIC_CODE_CHUNK_ACCEPTED',index=v['index'])))
'''
STAGE=r'''
lock=guard();data=b''.join((code/('part-%04d'%i)).read_bytes()for i in range(v['parts']));assert hashlib.sha256(data).hexdigest()==v['bundle_sha'];payload=json.loads(gzip.decompress(data));assert set(payload)==set(v['files'])
for name,text in payload.items():
 raw=base64.b64decode(text);item=v['files'][name];assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256'];p=code/name
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
pub=dict(commit=v['commit'],files={name:proof(code/name)for name in payload},code_root=str(code));p=b/'source-publication.json'
if p.exists():assert read(p)==pub
else:atomic(p,pub)
root=read(pathlib.Path(v['target'])/'source-publication.json');assert root['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e';runtime=pathlib.Path(root['code_root'])/'reproduction/operations/qwen7b_warmup/runtime.sh';env=dict(os.environ,QWEN7B_CODE_ROOT=root['code_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
fixture="import importlib.util,sys,unittest; import reproduction.operations.qwen7b_warmup as pkg; from pathlib import Path; root=Path("+repr(str(code))+"); name='reproduction.operations.qwen7b_warmup.direct_count_prior_decision'; spec=importlib.util.spec_from_file_location(name,root/'direct_count_prior_decision.py'); m=importlib.util.module_from_spec(spec); sys.modules[name]=m;spec.loader.exec_module(m);pkg.direct_count_prior_decision=m;spec=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.test_direct_count_prior_decision',root/'test_direct_count_prior_decision.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(t));assert r.wasSuccessful()"
r=subprocess.run(['bash',str(runtime),'-c',fixture],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,env=env,timeout=25);(b/'fixtures.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-1600:]
print(json.dumps(dict(status='TWO_DECISION_CPU_SOURCE_AND_TWO_FIXTURES_PASSED',source=proof(p),config=proof(code/'config.json'),fixtures=proof(b/'fixtures.log'))))
'''
RUN=r'''
lock=guard();pub=read(b/'source-publication.json')
for item in pub['files'].values():assert proof(item['path'])==item
if (b/'summary.json').exists():print(json.dumps(dict(status='EXISTING_FINITE_SUMMARY_PRESERVED',summary=read(b/'summary.json'),receipt=proof(b/'summary.json'))))
else:
 assert not (b/'intent.json').exists(),'Unknown prior outcome; preserve partial, never repeat blindly'
 atomic(b/'intent.json',dict(source=proof(b/'source-publication.json'),epoch=time.time(),GPU=0))
 root=read(pathlib.Path(v['target'])/'source-publication.json');env=dict(os.environ,QWEN7B_CODE_ROOT=root['code_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (b/'cpu.log').open('ab')as log:r=subprocess.run(['bash',root['code_root']+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u',str(code/'direct_count_prior_decision.py'),'--config',str(code/'config.json'),'--output',str(b)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=48)
 atomic(b/'exit.json',dict(returncode=r.returncode,epoch=time.time()));assert r.returncode==0,(b/'cpu.log').read_text()[-1800:]
 print(json.dumps(dict(status='ACTUAL_FULL938_TWO_DECISION_CPU_DIAGNOSTIC_COMPLETE',summary=read(b/'summary.json'),receipt=proof(b/'summary.json'),exit=read(b/'exit.json'))))
'''
STATUS=r'''
out=dict(epoch=time.time(),progress=read(b/'progress.json')if (b/'progress.json').exists()else None,exit=read(b/'exit.json')if (b/'exit.json').exists()else None)
if (b/'summary.json').exists():out.update(summary=read(b/'summary.json'),receipt=proof(b/'summary.json'))
print(json.dumps(out))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--action',choices=('stage','run','status'),required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3];config_name=str(a.config.resolve().relative_to(repo));files={};payload={}
 for name in FILES+[config_name]:
  raw=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo);assert raw==(repo/name).read_bytes();short='config.json'if name==config_name else Path(name).name;payload[short]=base64.b64encode(raw).decode();files[short]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 values=dict(base=BASE,target=TARGET,commit=a.commit,files=files);blocks=dict(chunk=CHUNK,stage=STAGE,run=RUN,status=STATUS)
 spec=importlib.util.spec_from_file_location('count_prior_decision_private_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(action,extra=None):
  script='v='+repr(dict(values,**(extra or {})))+'\n'+COMMON+'\n'+blocks[action];compile(script,action,'exec');command='python3 -c '+shlex.quote(script);assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=55);rows=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines()if x.startswith('{')]
  if r.returncode or not rows:raise RuntimeError(r.stdout.decode(errors='replace')[-1800:]+r.stderr.decode(errors='replace')[-1200:])
  return rows[-1]
 if a.action=='stage':
  remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0];assert remote==a.commit;blob=gzip.compress(json.dumps(payload,sort_keys=True).encode(),mtime=0)
  for i,start in enumerate(range(0,len(blob),5000)):
   part=blob[start:start+5000];call('chunk',dict(index=i,part=base64.b64encode(part).decode(),part_sha=hashlib.sha256(part).hexdigest()))
  answer=call('stage',dict(parts=(len(blob)+4999)//5000,bundle_sha=hashlib.sha256(blob).hexdigest()))
 else:answer=call(a.action)
 out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('full938-prior-decision-'+a.action+'.json');out.write_text(json.dumps(answer,indent=2,sort_keys=True)+'\n');print(json.dumps(answer))
if __name__=='__main__':main()
