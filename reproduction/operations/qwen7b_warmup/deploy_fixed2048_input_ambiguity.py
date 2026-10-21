"""One bounded CPU metadata audit; never submits GPU or changes training."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
from .deploy_hpc3_tail_g5 import remote
R=Path(__file__).resolve().parents[3]
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/fixed2048-input-ambiguity'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/fixed2048-input-ambiguity-v1'
FILES={'audit.py':R/'reproduction/operations/qwen7b_warmup/audit_fixed2048_input_ambiguity.py','input-packet.json':REPORT/'INPUT-PACKET.json'}

def run(commit,action):
    commit=subprocess.check_output(['git','rev-parse',commit],cwd=R).decode().strip()
    if not any(line.split()[0]==commit for line in subprocess.check_output(['git','ls-remote','origin'],cwd=R).decode().splitlines()):raise ValueError('Audit source must be published with exact remote ref')
    own=Path(__file__);assert subprocess.check_output(['git','show',commit+':'+str(own.relative_to(R))],cwd=R)==own.read_bytes()
    refs={};code_root=BASE+'/code-'+commit[:12]
    common='base='+repr(BASE)+'\nroot='+repr(code_root)+'\ncommit='+repr(commit)+'\n'+'''import pathlib,json,hashlib,base64,os,time
b=pathlib.Path(base);code=pathlib.Path(root)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw,'Immutable staged audit differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
    if action=='stage':
        for name,path in FILES.items():
            raw=path.read_bytes();assert subprocess.check_output(['git','show',commit+':'+str(path.relative_to(R))],cwd=R)==raw
            ref=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());refs[name]=dict(path=str(path.relative_to(R)),**ref)
            for i in range(0,len(raw),4096):
                part=raw[i:i+4096]
                remote(common+'v='+repr(dict(name=name,index=i//4096,data=base64.b64encode(part).decode(),sha256=hashlib.sha256(part).hexdigest()))+'\n'+'''raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256'];once(code/(v['name']+'.part-%04d'%v['index']),raw);print(json.dumps(dict(staged_part=v['name'],part=v['index'])))''')
            remote(common+'v='+repr(dict(name=name,parts=(len(raw)+4095)//4096,**ref))+'\n'+'''raw=b''.join((code/(v['name']+'.part-%04d'%i)).read_bytes()for i in range(v['parts']));assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['sha256'];once(code/v['name'],raw);print(json.dumps(dict(staged=proof(code/v['name']))))''')
        result=remote(common+'refs='+repr(refs)+'\n'+'''pub=dict(commit=commit,code_root=root,files=refs,CPU_only=True,GPU_submitted=False);once(code/'source-publication.json',(json.dumps(pub,indent=2,sort_keys=True)+'\\n').encode());print(json.dumps(dict(status='IMMUTABLE_CPU_METADATA_AUDIT_STAGED',source_publication=proof(code/'source-publication.json'),epoch=time.time())))''')
    elif action=='run':
        result=remote(common+'''import fcntl,subprocess,sys
pub=json.loads((code/'source-publication.json').read_text());assert pub['commit']==commit
for name,ref in pub['files'].items():
 actual=proof(code/name);assert actual['bytes']==ref['bytes']and actual['sha256']==ref['sha256']
lock=(b/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);done=b/'complete.json'
if done.exists():
 print(json.dumps(dict(status='EXISTING_IMMUTABLE_AUDIT_FRESH_READBACK',receipt=proof(done),result=json.loads(done.read_text()),epoch=time.time())))
else:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 r=subprocess.run([sys.executable,str(code/'audit.py'),'--packet',str(code/'input-packet.json'),'--output',str(done)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=env,timeout=80)
 once(b/'actual-audit-stdout.log',r.stdout.encode());once(b/'actual-audit-stderr.log',r.stderr.encode())
 if r.returncode:raise RuntimeError('CPU metadata audit failed: '+r.stderr[-2500:])
 print(json.dumps(dict(status='BOUNDED_CPU_AUDIT_EXITED',exit_code=r.returncode,receipt=proof(done),result=json.loads(done.read_text()),epoch=time.time())))
''',timeout=95)
    else:
        result=remote(common+'''p=b/'complete.json';out=dict(status='WAITING_FOR_EXISTING_AUDIT',exists=p.exists(),epoch=time.time())
if p.exists():out.update(status='EXISTING_AUDIT_FRESH_SHA_READBACK',receipt=proof(p),result=json.loads(p.read_text()))
print(json.dumps(out))''')
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/('actual-'+action+'.json')).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--action',choices=['stage','run','status'],required=True);a=p.parse_args();run(a.commit,a.action)
