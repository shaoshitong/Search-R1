"""Publish/stage/run the bounded metadata join; no training/GPU or payload relay."""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from .deploy_confirmation_materialization import import_closure

R = Path(__file__).resolve().parents[3]
CONTROL = R.parents[1]
PKG = 'reproduction/operations/qwen7b_warmup/'
A = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
BASE = A + '/joint124194-native-training-bank-v1'
REPORT = R / 'reproduction/reports/qwen7b-algorithm-improvement-20261021/joint124194-root-latest'
TEMPLATE = dict(role='fixed971_four_real_G10_banks_native_joint', target=BASE, questions=124194,
    source_frozen_batches=971, domains=dict(nq=78871, hotpotqa=45323),
    canonical_train_repeated_text_questions=3,
    evaluation_exclusions=dict(path=A+'/joint-balanced-native-E4-qualification-v3/evaluation-identity-exclusions.json',
        bytes=847206, sha256='a4ed20a3732984d13a099494d7bbb599aeb05b2868945b4f1f450786963b9b0a'),
    banks=[
        dict(key='existing58432', questions=58432, complete=dict(path=A+'/peer-threebank58432-internal5120-plan-v1/complete.json'),
             expected_complete_sha256='8a4ed0b24296ad8a5edc1bcecb249565b877d3bababb3f0221a8e7c26da72d06'),
        dict(key='recovered1216', questions=1216, complete=dict(path=A+'/peer-joint124194-recovered1216-L26-v1/bank/complete.json'),
             expected_complete_sha256='cedfa2e1892b03803182ec38a3d1c8d159167a62294e646858c3563d0942672a'),
        dict(key='audited243', questions=31010, source_batches=243,
             receiver_snapshot='audited243-after466-20261021-45227948',
             source_complete_sha256='a30b663dc9a4596718a6e1a3ff59ecb8e39f29efc16f94b1c45a9ee825044921',
             complete=dict(path=A+'/peer-joint124194-audited243-L26-v1/bank/complete.json')),
        dict(key='audited262', questions=33536, source_batches=262,
             receiver_snapshot='audited262-after709-20261021-d3c45767',
             source_complete_sha256='77c88160d009e331a2ccdfa1c632554c52e021c18b173bbf07c0cb5a2259c6be',
             complete=dict(path=A+'/peer-joint124194-audited262-L26-v1/bank/complete.json'))])

COMMON = '''import pathlib,json,hashlib,base64,gzip,tarfile,io,subprocess,sys,os,time,fcntl
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def immutable(p,x):
 if p.exists():
  if read(p)!=x:raise ValueError('Conflicting immutable code/config '+str(p))
 else:
  tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,sort_keys=True,separators=(',',':'))+'\\n');os.replace(str(tmp),str(p))
'''


def remote(script, timeout=45):
    # Only source/config is written locally. Credentials stay in existing control RAM.
    with tempfile.NamedTemporaryFile(prefix='q7b-join-control-', suffix='.py', mode='w', delete=False) as f:
        f.write(script); name = f.name
    try:
        cmd = ['python3', str(CONTROL/'outputs/qwen7b-warmup-20261019/balanced-control.py'), name, '--timeout', str(timeout)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout+10)
        rows = [json.loads(line) for line in result.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
        if result.returncode or not rows:
            raise RuntimeError('Metadata action unconfirmed; reconcile intent before retry: '+result.stderr.decode(errors='replace')[-1000:])
        return rows[-1]
    finally:
        os.unlink(name)


def bundle(commit):
    paths = [str(p.relative_to(R)) for p in (R/'reproduction').rglob('*.py') if '__pycache__' not in p.parts]
    names, _, missing = import_closure(paths, lambda n: (R/n).read_bytes(),
        seeds=(PKG+'joint124194_bank.py',), resource_files=())
    if missing: raise ValueError('Missing exact source closure '+str(missing))
    output = io.BytesIO(); refs = {}
    with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            for name in names:
                data = subprocess.check_output(['git', 'show', commit+':'+name], cwd=R)
                if data != (R/name).read_bytes(): raise ValueError('Unpublished metadata runtime '+name)
                info = tarfile.TarInfo(name); info.size=len(data); info.mode=0o444; info.mtime=0
                archive.addfile(info, io.BytesIO(data))
                refs[name]=dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return output.getvalue(), refs


def action(kind, commit):
    common = 'v='+repr(dict(base=BASE, commit=commit, pkg=PKG))+'\n'+COMMON
    if kind == 'stage':
        data, refs = bundle(commit); total=(len(data)+5999)//6000
        for i in range(total):
            part=data[6000*i:6000*(i+1)]
            script=common+'part='+repr(dict(i=i, data=base64.b64encode(part).decode(), sha256=hashlib.sha256(part).hexdigest()))+'\n'+'''
x=base64.b64decode(part['data']);assert hashlib.sha256(x).hexdigest()==part['sha256'];p=code/('part-%04d.bin'%part['i'])
if p.exists():assert p.read_bytes()==x
else:p.write_bytes(x)
print(json.dumps(dict(staged_part=part['i'],bytes=len(x))))
'''
            remote(script)
        return remote(common+'spec='+repr(dict(parts=total, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), files=refs))+'\n'+'''
payload=b''.join((code/('part-%04d.bin'%i)).read_bytes()for i in range(spec['parts']));assert len(payload)==spec['bytes']and hashlib.sha256(payload).hexdigest()==spec['sha256']
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz')as archive:
 for member in archive:
  assert member.name in spec['files']and member.isfile();data=archive.extractfile(member).read();ref=spec['files'][member.name];assert len(data)==ref['bytes']and hashlib.sha256(data).hexdigest()==ref['sha256'];p=code/member.name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==data
  else:p.write_bytes(data);p.chmod(0o444)
immutable(b/'source-publication.json',dict(commit=v['commit'],code_root=str(code),files=spec['files'],no_GPU=True,local_payload_transfer=False))
r=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_warmup.joint124194_bank','--help'],cwd=str(code),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
if r.returncode:raise ValueError('Published stdlib metadata entry failed: '+r.stderr[-1000:])
print(json.dumps(dict(status='FIXED124194_METADATA_JOIN_SOURCE_STAGED',publication=proof(b/'source-publication.json'),code_files=len(spec['files']))))
''')
    if kind == 'prepare':
        return remote(common+'template='+repr(TEMPLATE)+'\n'+'''
missing=[x['complete']['path']for x in template['banks']if not pathlib.Path(x['complete']['path']).exists()]
if missing:print(json.dumps(dict(status='WAIT_FOR_ACTUAL_SUBSET_COMPLETIONS',missing=missing,GPU_submitted=False)))
else:
 for item in template['banks']:
  item['complete']=proof(item['complete']['path'])
  if item.get('expected_complete_sha256')and item['complete']['sha256']!=item['expected_complete_sha256']:raise ValueError('Previously frozen bank completion changed')
 immutable(b/'config.json',template);print(json.dumps(dict(status='FIXED124194_METADATA_JOIN_CONFIG_READY',config=proof(b/'config.json'),source=proof(b/'source-publication.json'))))
''')
    if kind == 'start':
        return remote(common+'''
guard=(b/'launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(b/'complete.json').exists():print(json.dumps(dict(status='ALREADY_COMPLETE',complete=proof(b/'complete.json'))));sys.exit(0)
if(b/'controller.json').exists():
 c=read(b/'controller.json');p=pathlib.Path('/proc')/str(c['pid'])
 if p.exists():
  actual=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19];cmd=p.joinpath('cmdline').read_bytes()
  if actual==c['birth']and str(b).encode()in cmd:print(json.dumps(dict(status='ADOPT_EXISTING_JOIN',controller=c)));sys.exit(0)
 raise ValueError('Previous metadata worker is terminal; inspect failure/output before explicit same-config recovery')
config=proof(b/'config.json');pub=read(b/'source-publication.json');assert pub['commit']==v['commit']
for name,ref in pub['files'].items():assert proof(code/name)['sha256']==ref['sha256']
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
log=(b/'join.log').open('ab');p=subprocess.Popen([sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.joint124194_bank','--config',config['path'],'--config-sha256',config['sha256']],cwd=str(code),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
birth=pathlib.Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19];controller=dict(pid=p.pid,birth=birth,commit=v['commit'],config=config,started_epoch=time.time(),GPU=0,model_or_tensor_loaded=False);immutable(b/'controller.json',controller);print(json.dumps(dict(status='METADATA_JOIN_STARTED_NOT_COMPLETED',controller=controller)))
''')
    if kind == 'status':
        return remote(common+'''
out=dict(epoch=time.time(),base=str(b))
for n in ('controller','progress','complete','failure'):
 if(b/(n+'.json')).exists():out[n]=read(b/(n+'.json'))
if'controller'in out:
 c=out['controller'];p=pathlib.Path('/proc')/str(c['pid']);out['same_process_birth']=p.exists()and p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==c['birth']
if(b/'complete.json').exists():out['complete_ref']=proof(b/'complete.json')
print(json.dumps(out))
''')
    raise ValueError(kind)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--action', choices=('stage','prepare','start','status'), required=True)
    p.add_argument('--commit', required=True); p.add_argument('--branch', required=True); a=p.parse_args()
    actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+a.branch],cwd=R).decode().split()[0]
    if actual!=a.commit and subprocess.run(['git','merge-base','--is-ancestor',a.commit,actual],cwd=R).returncode:
        raise ValueError('Pinned join implementation not published on backup branch')
    own=str(Path(__file__).relative_to(R))
    if subprocess.check_output(['git','show',a.commit+':'+own],cwd=R)!=(R/own).read_bytes():
        raise ValueError('Unpublished join operator')
    result=action(a.action,a.commit);REPORT.mkdir(exist_ok=True)
    (REPORT/('bank-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
