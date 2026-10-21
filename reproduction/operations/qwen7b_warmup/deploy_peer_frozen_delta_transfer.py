"""Published code-only operator for the SHA-bound frozen raw delta migration.

Payload stays H100 -> management RAM -> private HF -> HKUST2 -> HKUST3 ->
HPC3. Only explicit stage/start/status actions; no GPU, producer writes, old
snapshot changes, automatic duplicate launches, or Mac payload staging.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
from . import deploy_peer_newseed_hf as hf
from . import deploy_peer_newseed_receive as recv
from . import deploy_peer_newseed_replay as ing
from . import peer_newseed_frozen_subset_export as subset

P='reproduction/operations/qwen7b_warmup/'
PLAN='frozen-selection.json'
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
CONFIG={};SNAPSHOT='';SOURCE=TARGET=INGEST=HPCDEP=None
SOURCE_FILES=()
def configure(config,config_name):
    global CONFIG,SNAPSHOT,SOURCE,TARGET,INGEST,HPCDEP,SOURCE_FILES
    CONFIG=dict(config);SNAPSHOT=CONFIG['snapshot']
    if not SNAPSHOT or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in SNAPSHOT) or not SNAPSHOT.startswith('audited'):raise ValueError('New explicit audited snapshot namespace required')
    if CONFIG.get('role')!='frozen_audited_delta_transfer' or CONFIG.get('no_source_discovery') is not True or CONFIG.get('no_GPU_or_training') is not True:raise ValueError('Explicit bounded metadata-only transfer config required')
    ref=CONFIG['source_selection']
    if not str(ref['path']).startswith('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/metadata-plans/') or len(ref['sha256'])!=64 or type(ref['bytes']) is not int or ref['bytes']<1:raise ValueError('Actual frozen canonical metadata plan descriptor required')
    SOURCE=Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021')/SNAPSHOT
    TARGET=B/'peer-newseed-h100-20261021'/SNAPSHOT
    INGEST=B/'algorithm-improvement-20261021'/('peer-newseed-ingest-'+SNAPSHOT)
    HPCDEP=B/'algorithm-improvement-20261021'/('peer-delta-import-runtime-'+SNAPSHOT)
    SOURCE_FILES=tuple(P+n for n in ('peer_newseed_frozen_subset_export.py','peer_newseed_export.py','peer_newseed_delta.py','peer_newseed_audit_reference.py','peer_newseed_source_audit_dispatch.py','peer_newseed_audit_resealed_metadata.py','test_peer_newseed_source_audit_dispatch.py','test_peer_newseed_frozen_subset_export.py'))+(config_name,)

ACTIONS=('source-stage','source-start','source-status','hf-stage','hf-start','hf-status','hf-first-archive','hf-resume-failed','receive-stage','receive-preflight','receive-start','receive-status','ingest-stage','ingest-tests','ingest-prepare','ingest-start','ingest-status')


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def sha(v):return hashlib.sha256(v).hexdigest()
def shell(argv):return shlex.join([str(x) for x in argv])


def publication(repo,commit,branch):
    actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]
    if actual!=commit and subprocess.run(['git','merge-base','--is-ancestor',commit,actual],cwd=repo).returncode:
        raise ValueError('Pinned source not in the actually published backup branch')
    return actual


def blob(repo,commit,name):return subprocess.check_output(['git','show',commit+':'+name],cwd=repo)


def bundle(repo,commit,names):
    output=io.BytesIO(); files={}
    with gzip.GzipFile(fileobj=output,mode='wb',mtime=0,filename='') as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as tar:
            for name in sorted(set(names)):
                raw=blob(repo,commit,name);files[name]=dict(bytes=len(raw),sha256=sha(raw))
                info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o444;tar.addfile(info,io.BytesIO(raw))
    return output.getvalue(),files


def json_call(argv,script=None,secret=None,timeout=55):
    if script is not None:
        if len(script.encode())>=16000:raise ValueError('Code control payload must stay below16KB')
        payload=script.encode()
    else:payload=canonical(secret)+b'\n' if secret is not None else None
    result=subprocess.run(argv,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    rows=[json.loads(x) for x in result.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
    if result.returncode or not rows:
        # These channels only carry source/code/control, not secrets in argv.
        raise RuntimeError('Action unconfirmed; reconcile same intent before retry: '+result.stderr.decode(errors='replace')[-1500:]+result.stdout.decode(errors='replace')[-1500:])
    return rows[-1]


def ctl_module():
    spec=importlib.util.spec_from_file_location('frozen_delta_existing_route','/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);return ctl


def source_call(script):return json_call(ctl_module().route('h100-10','python3 -'),script=script)


def source_stage(repo,commit):
    raw,files=bundle(repo,commit,SOURCE_FILES);root=str(SOURCE.parent/('_code-frozen-delta-'+SNAPSHOT+'-'+commit[:12]))
    for i in range(0,len(raw),6000):
        part=raw[i:i+6000];v=dict(root=root,offset=i,total=len(raw),sha=sha(raw),part=base64.b64encode(part).decode(),part_sha=sha(part))
        source_call('v='+repr(v)+'\n'+r'''
import base64,hashlib,json,pathlib,os,socket
assert socket.gethostname().split('.')[0]=='h100-10'
b=pathlib.Path(v['root']);b.mkdir(parents=True,exist_ok=True);p=b/'bundle.gz.upload';x=base64.b64decode(v['part']);assert hashlib.sha256(x).hexdigest()==v['part_sha']
n=p.stat().st_size if p.exists() else 0
if n==v['offset']:
 with p.open('ab') as f:f.write(x);f.flush();os.fsync(f.fileno())
else:
 assert n>=v['offset']+len(x)
 with p.open('rb') as f:f.seek(v['offset']);assert f.read(len(x))==x
print(json.dumps(dict(status='PUBLISHED_METADATA_CODE_CHUNK_ACCEPTED',offset=v['offset'],bytes=len(x))))
''')
    values=dict(root=root,files=files,bundle_sha=sha(raw),bundle_bytes=len(raw),commit=commit,source_selection=CONFIG['source_selection'],expected_new=CONFIG['expected_new'],expected_prior=CONFIG['expected_prior'],record_sha256=CONFIG['selected_batch_records_sha256'])
    return source_call('v='+repr(values)+'\n'+r'''
import hashlib,io,json,pathlib,tarfile,os,socket
assert socket.gethostname().split('.')[0]=='h100-10'
b=pathlib.Path(v['root']);raw=(b/'bundle.gz.upload').read_bytes();assert len(raw)==v['bundle_bytes'] and hashlib.sha256(raw).hexdigest()==v['bundle_sha']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 assert {m.name for m in tar.getmembers()}==set(v['files'])
 for m in tar.getmembers():
  p=pathlib.Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  x=tar.extractfile(m).read();info=v['files'][m.name];assert len(x)==info['bytes'] and hashlib.sha256(x).hexdigest()==info['sha256']
  p=b/p;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==x
  else:
   with p.open('xb') as f:f.write(x);f.flush();os.fsync(f.fileno())
   p.chmod(0o444)
ref=v['source_selection'];source=pathlib.Path(ref['path']);assert source.stat().st_size==ref['bytes'] and hashlib.sha256(source.read_bytes()).hexdigest()==ref['sha256'];target=b/'frozen-selection.json'
plan=json.loads(source.read_text());expected=v['expected_new'];assert all(plan[k]==expected[k] for k in ('batches','questions','trajectories')) and plan['domain_counts']==expected['domains'] and plan['batch_records_sha256']==v['record_sha256']
prior=v['expected_prior'];assert all(sum(p['expected_'+k] for p in plan['prior_snapshots'])==prior[k] for k in ('batches','questions','trajectories'))
if target.exists():assert target.read_bytes()==source.read_bytes()
else:target.write_bytes(source.read_bytes());target.chmod(0o444)
files=dict(v['files']);files['frozen-selection.json']=dict(bytes=ref['bytes'],sha256=ref['sha256'])
pub=dict(status='AUDITED_FROZEN_DELTA_SOURCE_CODE_AND_SELECTION_STAGED',commit=v['commit'],code_root=str(b),files=files)
p=b/'source-publication.json'
if p.exists():assert json.loads(p.read_text())==pub
else:p.write_text(json.dumps(pub,sort_keys=True))
print(json.dumps(dict(status=pub['status'],commit=v['commit'],code_root=str(b),files=len(v['files']))))
''')


def source_action(commit,action):
    code=str(SOURCE.parent/('_code-frozen-delta-'+SNAPSHOT+'-'+commit[:12]));v=dict(root=str(SOURCE),code=code,selection=code+'/'+PLAN,commit=commit,action=action,selection_sha256=CONFIG['source_selection']['sha256'])
    return source_call('v='+repr(v)+'\n'+r'''
import pathlib,json,hashlib,os,sys,time,fcntl,subprocess,socket
assert socket.gethostname().split('.')[0]=='h100-10'
b=pathlib.Path(v['root']);code=pathlib.Path(v['code']);op=b.parent/'_operators'/b.name;op.mkdir(parents=True,exist_ok=True)
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
def atomic(p,val):
 q=p.with_name(p.name+'.tmp-'+str(os.getpid()));q.write_text(json.dumps(val,sort_keys=True));q.replace(p)
def ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
pub=read(code/'source-publication.json');assert pub['commit']==v['commit']
for name,item in pub['files'].items():assert fp(code/name)==dict(item,path=str(code/name))
launch=op/'launch.json'
if v['action']=='source-start':
 with (op/'launch.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if launch.exists():
   old=read(launch);print(json.dumps(dict(status='EXISTING_SUBSET_EXPORT_NOT_REPLAYED',launch=old,actual_same_birth=ticks(old['pid'])==old['start_ticks'])));sys.exit(0)
  assert not (op/'launch-intent.json').exists(),'Unknown launch; reconcile before retry'
  assert not (b/'complete.json').exists() and not (b/'export.lock').exists(),'Unowned subset output requires reconciliation'
  env=dict(os.environ,PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
  test=subprocess.run([sys.executable,'-m','unittest','reproduction.operations.qwen7b_warmup.test_peer_newseed_frozen_subset_export'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=30)
  (op/'cpu-tests.log').write_text(test.stdout);assert test.returncode==0,test.stdout[-2000:]
  args=[sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.peer_newseed_frozen_subset_export','--selection',v['selection'],'--selection-sha256',v['selection_sha256'],'--snapshot-id',b.name]
  intent=dict(status='SUBSET_ONLY_EXPORT_INTENT',commit=v['commit'],selection=fp(pathlib.Path(v['selection'])),snapshot=str(b),epoch=time.time());atomic(op/'launch-intent.json',intent)
  with (op/'export.log').open('ab') as log:p=subprocess.Popen(args,env=env,cwd=str(code),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  receipt=dict(intent,status='SUBSET_EXPORT_STARTED_NOT_ADVANCEMENT_VERIFIED',pid=p.pid,start_ticks=ticks(p.pid));atomic(launch,receipt);print(json.dumps(receipt))
else:
 out=dict(status='READ_ONLY_FROZEN_DELTA_EXPORT_STATUS',epoch=time.time(),root=str(b))
 if launch.exists():
  val=read(launch);out['launch']=val;out['same_process_birth']=ticks(val['pid'])==val['start_ticks']
 for name in ('controller.json','subset-preflight-progress.json','failure.json'):
  if (b/name).exists():out[name]=read(b/name)
 if (b/'subset-complete.json').exists():out['subset_complete']=read(b/'subset-complete.json')
 if (b/'complete.json').exists():
  val=read(b/'complete.json');out['source_complete']=fp(b/'complete.json');out['complete']={k:val[k] for k in ('status','questions','trajectories','frozen_sealed_batches','files','payload_bytes','fresh_all_source_files_SHA256','all_archives_fresh_SHA_and_member_SHA_verified')};out['archives']=[{k:g[k] for k in ('group','archive','archive_bytes','archive_sha256')} for g in val['groups']]
 if not (b/'subset-complete.json').exists() and (op/'export.log').exists():
  with (op/'export.log').open('rb') as f:f.seek(max(0,(op/'export.log').stat().st_size-1600));out['log_tail']=f.read().decode(errors='replace')
 print(json.dumps(out))
''')


def mgmt_call(values,action,secret=None):
    script=hf.render(values,action);cmd='python3 -c '+shlex.quote(script)
    if len(cmd.encode())>=16000:raise ValueError('Management control command exceeds16KB')
    return json_call(['ssh','-T','-oBatchMode=yes','-oConnectTimeout=8','mgmt-host',cmd],secret=secret)


def hf_action(repo,commit,action):
    raw=blob(repo,commit,P+'peer_newseed_hf_stream.py');digest=sha(raw)
    v=dict(base=hf.PARENT+'/'+SNAPSHOT,source=str(SOURCE),snapshot=SNAPSHOT,code_sha256=digest,commit=commit,chunks=(len(raw)+4095)//4096)
    if action=='hf-stage':
        for i in range(v['chunks']):
            part=raw[i*4096:(i+1)*4096];mgmt_call(dict(v,index=i,data=base64.b64encode(part).decode(),chunk_sha256=sha(part)),'chunk')
        return mgmt_call(v,'stage')
    if action in ('hf-start','hf-resume-failed'):
        return mgmt_call(v,action[3:],dict(hf_token=recv.token()))
    if action=='hf-first-archive':
        script='import pathlib,json; b=pathlib.Path(%r); p=b/"uploaded/group-00000.json"; v=json.loads(p.read_text()); assert v["status"]=="PEER_SOURCE_PRIVATE_HF_ARCHIVE_SHA_ACCEPTED"; print(json.dumps(dict(status="ACTUAL_FIRST_PRIVATE_HF_ARCHIVE_SHA_BINDING",first_group={k:v[k] for k in ("hf_archive_path","hf_archive_commit","archive_bytes","archive_sha256")},catalog_finalized=(b/"complete.json").exists())))'%v['base']
        return json_call(['ssh','-T','-oBatchMode=yes','-oConnectTimeout=8','mgmt-host','python3 -c '+shlex.quote(script)])
    result=mgmt_call(v,'status')
    if 'complete' in result:
        script='import pathlib,json,hashlib; b=pathlib.Path(%r); v=json.loads((b/"complete.json").read_text()); g=v["groups"][0]; print(json.dumps(dict(catalog_commit=v["catalog_commit"],first_group={k:g[k] for k in ("hf_archive_path","hf_archive_commit","archive_bytes","archive_sha256")},source_questions=v["source_snapshot"]["questions"],source_trajectories=v["source_snapshot"]["trajectories"])))'%v['base']
        result['archive_binding']=json_call(['ssh','-T','-oBatchMode=yes','-oConnectTimeout=8','mgmt-host','python3 -c '+shlex.quote(script)])
    return result


def receive_action(repo,commit,args):
    raw=blob(repo,commit,P+'peer_newseed_receive.py')
    if raw!=Path(recv.receiver.__file__).read_bytes():raise ValueError('Imported receiver differs from published pin')
    base=recv.receiver.K2/SNAPSHOT;digest=sha(raw);code=base/'code'/digest/'peer_newseed_receive.py'
    v=dict(base=str(base),snapshot=SNAPSHOT,code=str(code),code_sha256=digest,source_commit=commit,prefix=recv.receiver.PREFIX+SNAPSHOT,catalog_commit=args.catalog_commit,hf_proxy=args.hf_proxy)
    with recv.route(False,False,args.via_hkust1) as ssh:
        if args.action=='receive-stage':
            actual=recv.receiver.install_code(ssh,base);assert actual==code
            return dict(status='PUBLISHED_INDEPENDENT_RECEIVER_CODE_STAGED',code=str(actual),sha256=digest,source_commit=commit)
        if args.action=='receive-preflight':
            if not args.archive_binding:raise ValueError('Actual uploaded first archive binding required')
            first=json.loads(args.archive_binding.read_bytes())['first_group']
            if not first['hf_archive_path'].startswith(v['prefix']+'/archives/'):raise ValueError('Other snapshot archive')
            return recv.receiver.remote(ssh,'python3 -c '+shlex.quote(recv.PREFLIGHT),dict(hf_token=recv.token(),archive_probe=first),timeout=55)
        secret=None
        if args.action=='receive-start':
            if not args.catalog_commit:raise ValueError('Actual catalog commit required')
            path=args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py';spec=importlib.util.spec_from_file_location('delta_private_control',str(path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
            secret=dict(hf_token=recv.token(),hpc3_password=m.password_for('mou176').decode())
        return recv.receiver.remote(ssh,'python3 -c '+shlex.quote(recv.render(v,args.action[8:])),secret,timeout=55)


def hpc_call(control,script,action):
    if len(('python3 -c '+shlex.quote(script)).encode())>=16000:raise ValueError('HPC3 control command exceeds16KB')
    folder=control/'outputs/qwen7b-warmup-20261019/peer-frozen-delta-transfer'/SNAPSHOT;folder.mkdir(parents=True,exist_ok=True);p=folder/('operator-'+action+'.py');p.write_text(script)
    r=subprocess.run(['python3',str(control/'outputs/qwen7b-warmup-20261019/balanced-control.py'),str(p),'--timeout','55'],capture_output=True,text=True,timeout=65)
    rows=[json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')]
    if r.returncode or not rows:raise RuntimeError('HPC3 action unconfirmed: '+r.stderr[-1200:]+r.stdout[-1200:])
    return rows[-1]


def ingest_action(repo,commit,args):
    raw,files=bundle(repo,commit,ing.FILES+((P+'peer_newseed_target_attestation.py',)if CONFIG.get('metadata_resealed_attestation_batches')else()))
    v=dict(base=str(HPCDEP),target=str(HPCDEP),receiver=str(TARGET),ingest=str(INGEST),commit=commit,bundle_sha256=sha(raw),bundle_bytes=len(raw),chunks=(len(raw)+4095)//4096,files=files,cpu_mode='ingest')
    def call(mode,extra=None):return hpc_call(args.control_root,ing.render(dict(v,**(extra or {})),mode),'ingest-'+mode)
    if args.action=='ingest-stage':
        call('init')
        for i in range(v['chunks']):
            part=raw[i*4096:(i+1)*4096];call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(part).decode(),chunk_sha256=sha(part)))
        return call('extract')
    return call({'ingest-tests':'cpu','ingest-prepare':'ingest-prepare','ingest-start':'start','ingest-status':'status'}[args.action])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--control-root',type=Path,required=True);parser.add_argument('--commit',required=True);parser.add_argument('--branch',required=True);parser.add_argument('--action',choices=ACTIONS,required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--catalog-commit');parser.add_argument('--archive-binding',type=Path);parser.add_argument('--hf-proxy');parser.add_argument('--via-hkust1',action='store_true');a=parser.parse_args()
    repo=Path(__file__).resolve().parents[3];publication(repo,a.commit,a.branch)
    config_path=a.config.resolve();name=str(config_path.relative_to(repo));data=blob(repo,a.commit,name)
    if data!=config_path.read_bytes():raise ValueError('Migration config differs from published immutable source')
    configure(json.loads(data),name)
    # Importing existing renderers never authorizes mutable or unpinned code.
    for module in (hf,recv,ing):
        relative=str(Path(module.__file__).relative_to(repo))
        if blob(repo,a.commit,relative)!=Path(module.__file__).read_bytes():raise ValueError('Renderer differs from published source '+relative)
    if a.action=='source-stage':result=source_stage(repo,a.commit)
    elif a.action.startswith('source-'):result=source_action(a.commit,a.action)
    elif a.action.startswith('hf-'):result=hf_action(repo,a.commit,a.action)
    elif a.action.startswith('receive-'):result=receive_action(repo,a.commit,a)
    else:result=ingest_action(repo,a.commit,a)
    result=dict(result,operator_source_commit=a.commit,snapshot=SNAPSHOT)
    out=a.control_root/'outputs/qwen7b-warmup-20261019/peer-frozen-delta-transfer'/SNAPSHOT;out.mkdir(parents=True,exist_ok=True)
    (out/(a.action+'.json')).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
