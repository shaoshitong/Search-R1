"""Publish-bound one-shot CPU metadata runner; no scheduler or scoring action."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
from .deploy_hpc3_tail_g5 import remote
R=Path(__file__).resolve().parents[3]
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/training-domain-role-priors'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/training-domain-role-priors-v1'
CONFIG='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1/config.json'
CONFIG_SHA='e5f0027dfcff4b6db442b38e9b2452e86cc2c9adcabd97a194971dc4d8f8d837'

def main(commit):
    commit=subprocess.check_output(['git','rev-parse',commit],cwd=R).decode().strip()
    if not any(line.split()[0]==commit for line in subprocess.check_output(['git','ls-remote','origin'],cwd=R).decode().splitlines()):raise ValueError('Exact source must be published')
    source=R/'reproduction/operations/qwen7b_warmup/audit_training_domain_role_priors.py';raw=source.read_bytes();own=Path(__file__)
    for path in(source,own):assert subprocess.check_output(['git','show',commit+':'+str(path.relative_to(R))],cwd=R)==path.read_bytes()
    code_root=BASE+'/code-'+commit[:12]
    common='base='+repr(BASE)+'\nroot='+repr(code_root)+'\n'+'''import pathlib,hashlib,json,base64,os,time
b=pathlib.Path(base);code=pathlib.Path(root)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw,'Immutable own audit artifact differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
    for i in range(0,len(raw),4096):
        part=raw[i:i+4096];remote(common+'part='+repr(dict(index=i//4096,data=base64.b64encode(part).decode(),sha256=hashlib.sha256(part).hexdigest()))+'\n'+'''raw=base64.b64decode(part['data']);assert hashlib.sha256(raw).hexdigest()==part['sha256'];once(code/('audit.part-%04d'%part['index']),raw);print(json.dumps(dict(part=part['index'],bytes=len(raw))))''')
    spec=dict(parts=(len(raw)+4095)//4096,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),commit=commit,config=CONFIG,config_sha256=CONFIG_SHA)
    result=remote(common+'v='+repr(spec)+'\n'+'''import fcntl,subprocess,sys
raw=b''.join((code/('audit.part-%04d'%i)).read_bytes()for i in range(v['parts']));assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['sha256'];once(code/'audit.py',raw)
once(code/'source-publication.json',(json.dumps(v,indent=2,sort_keys=True)+'\\n').encode())
lock=(b/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);done=b/'complete.json'
if not done.exists():
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 p=subprocess.run([sys.executable,str(code/'audit.py'),'--config',v['config'],'--config-sha256',v['config_sha256'],'--output',str(done)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=env,timeout=55)
 once(b/'actual-stdout.log',p.stdout.encode());once(b/'actual-stderr.log',p.stderr.encode())
 if p.returncode:raise RuntimeError(p.stderr[-2500:])
print(json.dumps(dict(status='ONE_SHOT_DOMAIN_ROLE_CPU_AUDIT_COMPLETE',source_publication=proof(code/'source-publication.json'),receipt=proof(done),result=json.loads(done.read_text()),epoch=time.time())))''',timeout=70)
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'actual-run.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    readback=remote(common+'''p=b/'complete.json';print(json.dumps(dict(status='FRESH_SMALL_RECEIPT_SHA_READBACK',receipt=proof(p),epoch=time.time())))''')
    assert result['receipt']==readback['receipt'];(REPORT/'actual-readback.json').write_text(json.dumps(readback,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);main(p.parse_args().commit)
