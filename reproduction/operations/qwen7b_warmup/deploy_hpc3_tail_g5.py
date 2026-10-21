"""Code/config-only tail preparation. Intentionally has NO GPU submit action."""
import argparse,base64,hashlib,importlib.util,json,os,shlex,subprocess,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1]
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail-newseedG5-20261021-v1'
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/hpc3-tail-newseed-preparation'
FILES={name:R/'reproduction/operations/qwen7b_warmup'/name for name in ['hpc3_tail_g5.py','hpc3_tail_raw_native_vendor.py','test_hpc3_tail_g5.py']}
FILES.update({name:REPORT/name for name in ['source-tail-plan.json','actual-HPC3-assets.json']})

def remote(code,timeout=55):
 if len(('python3 -c '+shlex.quote(code)).encode())>=16000:raise ValueError('Control command exceeds16KB')
 with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False)as f:f.write(code);p=Path(f.name)
 try:r=subprocess.run(['python3',str(CONTROL/'outputs/qwen7b-warmup-20261019/balanced-control.py'),str(p),'--timeout',str(timeout)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout+10)
 finally:p.unlink()
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1500:]+r.stdout.decode(errors='replace')[-2000:])
 rows=[json.loads(x)for x in r.stdout.decode().splitlines()if x.startswith('{')]
 if not rows:raise RuntimeError('No actual preparation receipt; do not blindly retry unknown mutation')
 return rows[-1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['stage','prepare','status'],required=True);p.add_argument('--commit');a=p.parse_args()
 commit=a.commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip()
 remotehead=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
 if commit!=remotehead and subprocess.run(['git','merge-base','--is-ancestor',commit,remotehead],cwd=R).returncode:raise ValueError('Exact preparation source must first be pushed')
 own=str(Path(__file__).relative_to(R))
 if subprocess.check_output(['git','show',commit+':'+own],cwd=R)!=Path(__file__).read_bytes():raise ValueError('Unpublished deployment entry')
 code_root=BASE+'/code-'+commit[:12]
 common='v='+repr(dict(base=BASE,code=code_root,commit=commit))+'\n'+'''import pathlib,json,hashlib,os,base64,subprocess,sys,time
b=pathlib.Path(v['base']);code=pathlib.Path(v['code'])
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw,'Immutable staged source/config differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
 if a.action=='stage':
  refs={}
  for name,path in FILES.items():
   rel=str(path.relative_to(R));raw=path.read_bytes()
   if subprocess.check_output(['git','show',commit+':'+rel],cwd=R)!=raw:raise ValueError('Unpublished tail source/config '+rel)
   refs[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
   for i in range(0,len(raw),4096):
    piece=raw[i:i+4096];remote(common+'part='+repr(dict(name=name,index=i//4096,data=base64.b64encode(piece).decode(),sha256=hashlib.sha256(piece).hexdigest()))+'\n'+'''data=base64.b64decode(part['data']);assert hashlib.sha256(data).hexdigest()==part['sha256'];once(code/(part['name']+'.part-%04d'%part['index']),data);print(json.dumps(dict(staged_file=part['name'],part=part['index'],bytes=len(data))))''')
   remote(common+'spec='+repr(dict(name=name,parts=(len(raw)+4095)//4096,**refs[name]))+'\n'+'''raw=b''.join((code/(spec['name']+'.part-%04d'%i)).read_bytes()for i in range(spec['parts']));assert len(raw)==spec['bytes']and hashlib.sha256(raw).hexdigest()==spec['sha256'];once(code/spec['name'],raw);print(json.dumps(dict(staged=proof(code/spec['name']))))''')
  result=remote(common+'refs='+repr(refs)+'\n'+'''publication=dict(commit=v['commit'],code_root=str(code),files=refs,GPU_submitted=False);once(b/'source-publication.json',(json.dumps(publication,indent=2,sort_keys=True)+'\\n').encode());print(json.dumps(dict(status='TAIL_INDEPENDENT_CODE_AND_METADATA_STAGED_NO_GPU',publication=proof(b/'source-publication.json'))))''')
 elif a.action=='prepare':
  result=remote(common+'''pub=read(b/'source-publication.json');assert pub['commit']==v['commit']
for name,ref in pub['files'].items():assert proof(code/name)['sha256']==ref['sha256']
assets=read(code/'actual-HPC3-assets.json');old=pathlib.Path(assets['qualified_code_root']);runtime=old/'reproduction/operations/qwen7b_warmup/runtime.sh'
r=subprocess.run([sys.executable,str(code/'test_hpc3_tail_g5.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=15)
assert r.returncode==0,r.stdout
once(b/'actual-static-tests.log',r.stdout.encode())
args=[sys.executable,str(code/'hpc3_tail_g5.py'),'--prepare','--source-plan',str(code/'source-tail-plan.json'),'--source-plan-sha256',proof(code/'source-tail-plan.json')['sha256'],'--assets',str(code/'actual-HPC3-assets.json'),'--assets-sha256',proof(code/'actual-HPC3-assets.json')['sha256'],'--source-commit',v['commit'],'--code-root',str(code)]
r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=120)
(b/'actual-prepare.log').write_text(r.stdout)
if r.returncode:raise RuntimeError(r.stdout[-2500:])
print(json.dumps(dict(status='TAIL_PREPARE_RETURNED',prepared=read(b/'prepared.json'),tests=proof(b/'actual-static-tests.log'),GPU_submitted=False)))''',timeout=145)
 else:
  result=remote(common+'''out=dict(epoch=time.time(),base=str(b),GPU_submit_action_available=False)
for name in ['source-publication.json','prepared.json']:
 if(b/name).exists():out[name]=read(b/name)
print(json.dumps(out))''')
 REPORT.mkdir(exist_ok=True);(REPORT/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
