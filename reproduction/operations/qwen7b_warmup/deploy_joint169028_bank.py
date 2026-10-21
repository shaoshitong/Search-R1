"""Publish/stage a source-bound five-bank metadata join; never submit a GPU.

Preparation waits for real producer acceptances. The one metadata worker runs
with two CPU cores in this task's already verified main allocation and keeps
all large metadata on shared storage. Existing124194 code/data stay frozen.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from .deploy_joint124194_bank import COMMON, remote
from .joint169028_bank import COUNTS, QID_SHA, STATUS

R = Path(__file__).resolve().parents[3]
PKG = 'reproduction/operations/qwen7b_warmup/'
A = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
BASE = A + '/joint169028-native-training-bank-v1'
REF = 'refs/heads/codex/q7b-native-joint-root-20261021'
REPORT = R / 'reproduction/reports/qwen7b-algorithm-improvement-20261021/full169028-native-preparation'
PATHS = dict(current124194=A+'/joint124194-native-training-bank-v1/complete.json',
    HPC3_first9344=A+'/hpc3-tail9344-independent-L26-prepared-v2/bank/complete.json',
    donor_only14080=A+'/donor14080-independent-L26-prepared-v1/bank/complete.json',
    HPC3_next14592=A+'/hpc3-tail14592-independent-L26-prepared-v1/bank/complete.json',
    HPC3_final6818=A+'/hpc3-tail6818-independent-L26-prepared-v1/bank/complete.json')
TEMPLATE = dict(role='full169028_sameHF_source_resolved_native_joint', target=BASE,
    questions=169028, canonical_qid_sha256=QID_SHA,
    canonical_groups=dict(path=A.rsplit('/',1)[0]+'/bank/groups.json', bytes=111281750,
        sha256='1dbec76bab828a391de66f01fdaabfafc1557f699df174e82d81952263278bf8'),
    full_raw_G10_acceptance=dict(path=A+'/hpc3-tail-full169028-qualified-raw-union-v1/complete.json',
        bytes=7943, sha256='3dabf439073b1c49b4b0be7e9885100a214f101443e3b7a983ddca1fbbf40b1b'),
    evaluation_exclusions=dict(path=A+'/joint-balanced-native-E4-qualification-v3/evaluation-identity-exclusions.json',
        bytes=847206, sha256='a4ed20a3732984d13a099494d7bbb599aeb05b2868945b4f1f450786963b9b0a'),
    banks=[dict(key=k, questions=n, complete=dict(path=PATHS[k])) for k,n in COUNTS.items()])


def bundle(commit):
    names = [PKG+n for n in ('joint169028_bank.py', 'joint124194_bank.py',
        'replay_probe_layers.py', 'replay_feature_reader.py', 'g10_grouping.py')]
    for folder in ('reproduction', 'reproduction/operations', PKG.rstrip('/')):
        if (R/folder/'__init__.py').exists():
            names.append(folder+'/__init__.py')
    output, refs = io.BytesIO(), {}
    with gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            for name in names:
                raw = subprocess.check_output(['git','show',commit+':'+name], cwd=R)
                if raw != (R/name).read_bytes():
                    raise ValueError('Unpublished full-data metadata runtime '+name)
                info = tarfile.TarInfo(name);info.size=len(raw);info.mode=0o444;info.mtime=0
                archive.addfile(info, io.BytesIO(raw))
                refs[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return output.getvalue(), refs


def action(kind, commit):
    common = 'v='+repr(dict(base=BASE,commit=commit,pkg=PKG,status=STATUS))+'\n'+COMMON
    if kind == 'stage':
        raw, refs = bundle(commit)
        for i,start in enumerate(range(0,len(raw),6000)):
            part=raw[start:start+6000]
            remote(common+'part='+repr(dict(i=i,data=base64.b64encode(part).decode(),sha=hashlib.sha256(part).hexdigest()))+'\n'+'''
x=base64.b64decode(part['data']);assert hashlib.sha256(x).hexdigest()==part['sha'];p=code/('part-%04d.bin'%part['i'])
if p.exists():assert p.read_bytes()==x
else:p.write_bytes(x)
print(json.dumps(dict(staged_part=part['i'],bytes=len(x))))
''')
        return remote(common+'spec='+repr(dict(parts=(len(raw)+5999)//6000,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),files=refs))+'\n'+'''
payload=b''.join((code/('part-%04d.bin'%i)).read_bytes()for i in range(spec['parts']));assert len(payload)==spec['bytes']and hashlib.sha256(payload).hexdigest()==spec['sha256']
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz')as archive:
 for member in archive:
  assert member.name in spec['files']and member.isfile();raw=archive.extractfile(member).read();f=spec['files'][member.name];assert len(raw)==f['bytes']and hashlib.sha256(raw).hexdigest()==f['sha256'];p=code/member.name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw);p.chmod(0o444)
immutable(b/'source-publication.json',dict(commit=v['commit'],code_root=str(code),files=spec['files'],GPU=0,model_or_tensor_loaded=False))
r=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_warmup.joint169028_bank','--help'],cwd=str(code),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
if r.returncode:raise ValueError('Actual published stdlib entry failed: '+r.stderr[-1500:])
print(json.dumps(dict(status='FULL169028_SOURCE_BOUND_METADATA_JOIN_STAGED',publication=proof(b/'source-publication.json'),files=len(spec['files']))))
''')
    if kind == 'prepare':
        return remote(common+'template='+repr(TEMPLATE)+'\n'+'''
missing=[x['complete']['path']for x in template['banks']if not pathlib.Path(x['complete']['path']).exists()]
if missing:print(json.dumps(dict(status='WAIT_FOR_ACTUAL_FULL169028_FEATURE_BANKS',missing=missing,GPU_submitted=False,config_written=False)))
else:
 def checked_ref(ref):
  assert proof(ref['path'])==ref,'Previously sealed actual metadata changed';return read(ref['path'])
 raw=checked_ref(template['full_raw_G10_acceptance']);assert raw['status']=='EXACT_CANONICAL_169028_TRAINING_QUESTIONS_FULL_REAL_RAW_G10_UNION_QUALIFIED'and raw['questions']==169028 and raw['trajectories']==1690280 and raw['chosen_ordered_qid_sha256']==template['canonical_qid_sha256']
 checked_ref(template['canonical_groups']);checked_ref(template['evaluation_exclusions'])
 for item in template['banks']:
  item['complete']=proof(item['complete']['path']);x=read(item['complete']['path']);assert x['questions']==item['questions']and x['trajectories']==item['questions']*10
  if item['key']=='current124194':assert item['complete']['sha256']=='a8fe2c2ea823087d50ac86b283eb361b51ae7ba5ba0571860193214338200801'
  if item['key']=='HPC3_first9344':assert item['complete']['sha256']=='7bad8bd1864193f1e44665899e648ae17f4e9d7a0cee715ce9a282bbb93aa8e2'
 pub=read(b/'source-publication.json');assert pub['commit']==v['commit']
 immutable(b/'config.json',template);print(json.dumps(dict(status='FULL169028_METADATA_JOIN_CONFIG_READY',config=proof(b/'config.json'),source=proof(b/'source-publication.json'),GPU_submitted=False)))
''')
    if kind == 'start':
        return remote(common+'''
guard=(b/'launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(b/'complete.json').exists():
 done=read(b/'complete.json');assert done['status']==v['status'];print(json.dumps(dict(status='ALREADY_COMPLETE',complete=proof(b/'complete.json'))));sys.exit(0)
if(b/'controller.json').exists():
 c=read(b/'controller.json');p=pathlib.Path('/proc')/str(c['pid'])
 if p.exists()and p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==c['birth']:
  assert b'joint169028_bank'in p.joinpath('cmdline').read_bytes();print(json.dumps(dict(status='ADOPT_EXISTING_FULL_JOIN',controller=c)));sys.exit(0)
 raise ValueError('Prior worker terminal: inspect actual failure and keep same config before recovery')
config=proof(b/'config.json');pub=read(b/'source-publication.json');assert pub['commit']==v['commit']
for name,ref in pub['files'].items():assert proof(code/name)['sha256']==ref['sha256']
job='745501';s=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True)
for token in ('JobState=RUNNING','UserId=mou176(21930)','Account=mou176','NumCPUs=8','WorkDir='+str(b.parent/'joint-balanced-native-E4-124194-v1'),'Command='+str(b.parent/'joint-balanced-native-E4-124194-v1/training.sbatch')):assert token in s,token
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
args=['srun','--overlap','--jobid='+job,'--account=mou176','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=4G','--gres=gpu:0',sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.joint169028_bank','--config',config['path'],'--config-sha256',config['sha256']]
log=(b/'join.log').open('ab');p=subprocess.Popen(args,cwd=str(code),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
birth=pathlib.Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19];controller=dict(pid=p.pid,birth=birth,commit=v['commit'],config=config,allocation_job=job,CPU=2,GPU=0,started_epoch=time.time());immutable(b/'controller.json',controller)
print(json.dumps(dict(status='FULL169028_METADATA_JOIN_STARTED_NOT_COMPLETED',controller=controller)))
''')
    if kind == 'status':
        return remote(common+'''
out=dict(epoch=time.time(),base=str(b))
for n in ('controller','worker','progress','complete','failure'):
 try:
  p=b/(n+'.json')
  if p.exists():out[n]=read(p)
 except FileNotFoundError:pass
for name in ('controller','worker'):
 if name in out:
  c=out[name];p=pathlib.Path('/proc')/str(c['pid']);out[name+'_same_birth']=p.exists()and p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==c['birth']
if(b/'complete.json').exists():out['complete_ref']=proof(b/'complete.json')
print(json.dumps(out))
''')
    raise ValueError(kind)


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','start','status'),required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    remote_sha=subprocess.check_output(['git','ls-remote','origin',REF],cwd=R,text=True).split()[0]
    if subprocess.run(['git','merge-base','--is-ancestor',a.commit,remote_sha],cwd=R).returncode:
        raise ValueError('Full-data join code not published on its exact backup branch')
    own=str(Path(__file__).relative_to(R))
    if subprocess.check_output(['git','show',a.commit+':'+own],cwd=R)!=(R/own).read_bytes():
        raise ValueError('Unpublished full-data metadata controller')
    result=action(a.action,a.commit);REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/('actual-bank-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
