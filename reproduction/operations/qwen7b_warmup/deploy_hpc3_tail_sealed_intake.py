"""One bounded CPU metadata intake, no generation/transfer/GPU submit action."""
import argparse,base64,hashlib,importlib.util,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=Path(__file__).parent
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/hpc3-tail-sealed-intake'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail-sealed-intake-v1'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('stage','prepare','status'),required=True);parser.add_argument('--commit');a=parser.parse_args()
 commit=a.commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();origin=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
 if commit!=origin and subprocess.run(['git','merge-base','--is-ancestor',commit,origin],cwd=R).returncode:raise ValueError('Source must be published before metadata preparation')
 for name in('hpc3_tail_sealed_intake.py','test_hpc3_tail_sealed_intake.py',Path(__file__).name):
  path=P/name
  if subprocess.check_output(['git','show',commit+':'+str(path.relative_to(R))],cwd=R)!=path.read_bytes():raise ValueError('Unpublished intake source '+name)
 spec=importlib.util.spec_from_file_location('transport',P/'deploy_hpc3_tail_g5.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 value=dict(base=BASE,code=BASE+'/code-'+commit[:12],commit=commit)
 common='V='+repr(value)+'\n'+'''import pathlib,json,hashlib,os,subprocess,sys,time
b=pathlib.Path(V['base']);code=pathlib.Path(V['code']);b.mkdir(exist_ok=True);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Immutable intake code differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
 if a.action=='stage':
  files={}
  for name in('hpc3_tail_sealed_intake.py','test_hpc3_tail_sealed_intake.py'):
   raw=(P/name).read_bytes();files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
   for i in range(0,len(raw),4096):
    chunk=raw[i:i+4096];t.remote(common+'part='+repr(dict(name=name,index=i//4096,data=base64.b64encode(chunk).decode()))+'\n'+'''import base64
raw=base64.b64decode(part['data']);once(code/(part['name']+'.part-%04d'%part['index']),raw);print(json.dumps(dict(part=part['index'],bytes=len(raw))))''')
   t.remote(common+'spec='+repr(dict(name=name,parts=(len(raw)+4095)//4096,**files[name]))+'\n'+'''raw=b''.join((code/(spec['name']+'.part-%04d'%i)).read_bytes()for i in range(spec['parts']));assert len(raw)==spec['bytes']and hashlib.sha256(raw).hexdigest()==spec['sha256'];compile(raw.decode(),spec['name'],'exec');once(code/spec['name'],raw);print(json.dumps(dict(staged=proof(code/spec['name']))))''')
  result=t.remote(common+'files='+repr(files)+'\n'+'''pub=dict(source_commit=V['commit'],code_root=str(code),files=files,model_or_tensor_loaded=False,submits_GPU=False);once(code/'source-publication.json',(json.dumps(pub,sort_keys=True)+'\\n').encode());print(json.dumps(dict(status='ISOLATED_INTAKE_CODE_SHA_STAGED',publication=proof(code/'source-publication.json'))))''')
 elif a.action=='prepare':
  result=t.remote(common+'''pub=read(code/'source-publication.json');assert pub['source_commit']==V['commit']
for name,ref in pub['files'].items():
 actual=proof(code/name);assert actual['sha256']==ref['sha256']and actual['bytes']==ref['bytes']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',INTAKE_SOURCE_COMMIT=V['commit'])
fixture=subprocess.run([sys.executable,str(code/'test_hpc3_tail_sealed_intake.py')],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=10);assert fixture.returncode==0,fixture.stdout;once(code/'synthetic-tests.log',fixture.stdout.encode())
p=subprocess.run([sys.executable,str(code/'hpc3_tail_sealed_intake.py'),'--prepare-sealed-metadata'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=145);log=code/('actual-prepare-%d.log'%int(time.time()));once(log,(p.stdout+p.stderr).encode())
if p.returncode:raise RuntimeError('Bounded metadata intake failed; producer unchanged: '+p.stderr[-1800:])
rows=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')];assert rows;print(json.dumps(dict(preparation=rows[-1],source_publication=proof(code/'source-publication.json'),tests=proof(code/'synthetic-tests.log'),log=proof(log)),sort_keys=True))''',timeout=160)
 else:
  result=t.remote(common+'''rows=[]
for path in sorted(b.glob('snapshot-*/complete.json')):
 value=read(path);rows.append(dict(complete=proof(path),status=value['status'],questions=value['sealed_questions'],future_eligible_questions=value['future_eligible_questions']))
print(json.dumps(dict(status='READ_ONLY_SNAPSHOT_STATUS',snapshots=rows)))''')
 REPORT.mkdir(parents=True,exist_ok=True);out=REPORT/('actual-'+a.action+'.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
