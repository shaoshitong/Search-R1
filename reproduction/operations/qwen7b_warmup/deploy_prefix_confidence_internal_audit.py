"""Code-only stage/run/status for one independent INTERNAL CPU scalar audit."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
from .deploy_confirmation_materialization import make_bundle

P='reproduction/operations/qwen7b_warmup/'
EXPERIMENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1'
TARGET=EXPERIMENT+'/independent-internal-audit-deployment-v1'
CONFIG_SHA='3ec15b790d5b0db67b88fdc22e74ea692eb216e34e0955853f4c273a30392371'
FIT_PUB_SHA='5a020457f2a0309afa6745946253b232b4aeb76d5400e51b3ae419069076d5d2'
INTERNAL_SHA='45de5e01ff1802f27acba1c25d7393865e281e8c53da673f24a2de73312086ed'
COMMON=r'''
import base64,fcntl,hashlib,io,json,os,pathlib,subprocess,tarfile,time
b=pathlib.Path(values['base']);b.mkdir(exist_ok=True);experiment=b.parent
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,v):
 p=pathlib.Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,sort_keys=True)+'\n');q.replace(p)
def proof(p):
 p=pathlib.Path(p);data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
guard=(b/'operator.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control-root',type=Path,required=True)
    parser.add_argument('--action',choices=('stage','run','status'),required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[3];spec=importlib.util.spec_from_file_location('prefix_internal_private_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(body,extra=None):
        values=dict(base=TARGET);values.update(extra or {});command='python3 -c '+shlex.quote('values='+repr(values)+'\n'+COMMON+'\n'+body)
        if len(command.encode())>=16000:raise ValueError('Internal-audit control command exceeds16KB')
        r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
        rows=[json.loads(v) for v in r.stdout.decode(errors='replace').splitlines() if v.startswith('{')]
        if r.returncode or not rows:raise RuntimeError('Independent CPU audit outcome unconfirmed; inspect same intent before retry: '+r.stderr.decode(errors='replace')[-1200:]+r.stdout.decode(errors='replace')[-1800:])
        return rows[-1]
    if args.action=='stage':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
        if remote!=commit:raise ValueError('Publish current auditor source before stage')
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=repo,text=True).splitlines();cache={}
        def load(name):
            if name not in cache:cache[name]=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
            return cache[name]
        payload,manifest=make_bundle(paths,load,commit,seeds=(P+'audit_prefix_confidence_internal.py',P+'test_audit_prefix_confidence_internal.py'),resource_files=(P+'runtime.sh',),manifest_name='_source.json')
        for i,start in enumerate(range(0,len(payload),4096)):
            block=payload[start:start+4096]
            call("root=b/('code-'+values['commit'][:12]);root.mkdir(exist_ok=True);p=root/('part-%04d'%values['index']);x=base64.b64decode(values['data']);assert hashlib.sha256(x).hexdigest()==values['sha256']\nif p.exists():assert p.read_bytes()==x\nelse:p.write_bytes(x)\nprint(json.dumps(dict(part=values['index'])))",dict(commit=commit,index=i,data=base64.b64encode(block).decode(),sha256=hashlib.sha256(block).hexdigest()))
        result=call(r'''
root=b/('code-'+values['commit'][:12]);data=b''.join((root/('part-%04d'%i)).read_bytes() for i in range(values['chunks']));assert hashlib.sha256(data).hexdigest()==values['sha256']
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 m=json.loads(tar.extractfile('_source.json').read());assert m['commit']==values['commit'];files=m['files'];assert {x.name for x in tar.getmembers()}==set(files)|{'_source.json'}
 for member in tar.getmembers():
  rel=pathlib.Path(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts;raw=tar.extractfile(member).read();p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if member.name in files:assert len(raw)==files[member.name]['bytes'] and hashlib.sha256(raw).hexdigest()==files[member.name]['sha256']
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw);p.chmod(0o444)
publication=dict(commit=values['commit'],remote_sha=values['commit'],code_root=str(root),files=files)
p=b/'source-publication.json'
if p.exists():assert read(p)==publication
else:atomic(p,publication)
r=subprocess.run(['python3','-m','unittest','reproduction.operations.qwen7b_warmup.test_audit_prefix_confidence_internal','-v'],cwd=str(root),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=15);(b/'stdlib-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-1500:]
print(json.dumps(dict(status='INDEPENDENT_INTERNAL_AUDITOR_CODE_AND_STDLIB_FIXTURES_STAGED',publication=proof(b/'source-publication.json'),code_root=str(root),tests=proof(b/'stdlib-tests.log'),actual_scalar_CP_opened=False,external_opened=False)))
''',dict(commit=commit,chunks=(len(payload)+4095)//4096,sha256=hashlib.sha256(payload).hexdigest()))
    elif args.action=='run':
        result=call(r'''
config=experiment/'fit-config-v2.json';pub=experiment/'fit-deployment-v2/source-publication.json';internal=experiment/'fit/internal-complete.json';output=experiment/'fit/independent-internal-audit.json'
assert proof(config)['sha256']==values['config_sha'] and proof(pub)['sha256']==values['pub_sha']
if output.exists():print(json.dumps(dict(status='EXISTING_INDEPENDENT_INTERNAL_AUDIT_PRESERVED',result=read(output),receipt=proof(output))))
elif not internal.exists():print(json.dumps(dict(status='WAITING_FOR_INTERNAL_COMPLETE_NO_SCALAR_CP_OPENED')))
else:
 assert proof(internal)['sha256']==values['internal_sha'];assert not (b/'intent.json').exists(),'Unknown prior audit: inspect same log/exit before repeat';atomic(b/'intent.json',dict(config=proof(config),internal_complete=proof(internal),CPU=2,GPUs=0,time=time.time()))
 source=read(b/'source-publication.json');root=pathlib.Path(source['code_root'])
 for name,v in source['files'].items():assert proof(root/name)['sha256']==v['sha256']
 wrapper="""import os,resource,runpy,threading,time,pathlib
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_DATA,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
def watch():
 while True:
  time.sleep(.1);rss=int(next(v for v in pathlib.Path('/proc/self/status').read_text().splitlines() if v.startswith('VmRSS:')).split()[1])*1024
  if rss>3*1024**3:os._exit(74)
threading.Thread(target=watch,daemon=True).start()
runpy.run_module('reproduction.operations.qwen7b_warmup.audit_prefix_confidence_internal',run_name='__main__')"""
 argv=['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-c',wrapper,'--config',str(config),'--config-sha256',values['config_sha'],'--source-publication',str(pub),'--source-publication-sha256',values['pub_sha'],'--internal-complete-sha256',values['internal_sha']]
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (b/'cpu.log').open('ab') as log:r=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
 atomic(b/'exit.json',dict(returncode=r.returncode,time=time.time()));assert r.returncode==0,(b/'cpu.log').read_text()[-2500:]
 if output.exists():print(json.dumps(dict(status='ACTUAL_INDEPENDENT_INTERNAL_AUDIT_COMPLETE',result=read(output),receipt=proof(output))))
 else:print(json.dumps(dict(status='AUDITOR_RETURNED_WAIT_NO_CP_OPENED',exit=read(b/'exit.json'))))
''',dict(config_sha=CONFIG_SHA,pub_sha=FIT_PUB_SHA,internal_sha=INTERNAL_SHA))
    else:
        result=call("p=experiment/'fit/independent-internal-audit.json';out=dict(receipt=proof(p),result=read(p)) if p.exists() else dict(status='NOT_YET_INDEPENDENTLY_ACCEPTED');out['exit']=read(b/'exit.json') if (b/'exit.json').exists() else None;print(json.dumps(out))")
    out=repo/'reproduction/reports/qwen7b-algorithm-improvement-20261021/prefix-confidence';out.mkdir(exist_ok=True)
    (out/('independent-internal-'+args.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
