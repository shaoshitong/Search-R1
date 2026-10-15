"""Code-only publication and remote-to-remote short-input transport.

No model, rollout, feature or prediction bytes pass through this Mac process.
The temporary forwarded SSH agent supplies signatures to the trusted management
host; private key bytes and global SSH configuration are never copied/changed.
"""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[2]
PROJECT = Path('/Users/shaoshitong/Documents/Search-G1')
MODEL_A800 = '/data/search-g1/assets/qwen2.5-3b-instruct'
MODEL_H100 = '/dataset/shaoshitong/Search-G1-common8/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725'
ENV_H100 = '/dataset/shaoshitong/Search-G1-common8/env'
RUNTIME_A800 = '/data/shaoshitong/Search-G1-a800-fresh200-16034e59031e4f21a323ebb2556bef7f/repo'
A800_USER = 'job-20e13974-c420-4074-a464-68e93738cfc0-master-0.ldwang.baai-sailing_aquila.dx-calc1.job'


def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args])


def a800_route(code):
    path = PROJECT/'qprobe-qwen-native-selection-metrics-fix/reproduction/a800_common/deploy.py'
    spec = importlib.util.spec_from_file_location('existing_a800_transport', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.ssh_command(code)


def record(args, role, value):
    path = Path(args.receipts)/(role+'-'+str(time.time_ns())+'.json')
    with path.open('x') as stream: json.dump(value,stream,indent=2)
    print(json.dumps(dict(receipt=str(path),**value)),flush=True)


def h100(args, code, role):
    from reproduction.h1008_qprobe.management_transport import remote
    socket = json.loads(Path(args.management_receipt).read_text())['socket']
    path = Path(args.receipts)/(role+'-transport-'+str(time.time_ns())+'.json')
    out = remote(code,path,control_path=socket)
    value=json.loads(out); record(args,role,value); return value


def export_inputs(args):
    raw=git('show',args.commit+':reproduction/prefix_only_correctness_ab/features.py')
    metadata=json.loads(Path(args.reference_metadata).read_text())
    t=metadata['teacher']; q=metadata['test']
    sources=dict(model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
        model_files_sha256=t['identity.json']['summary']['model_files_sha256'],
        train=dict(teacher_cache=str(Path(t['identity.json']['path']).parent),
            identity_sha256=t['identity.json']['sha256'],complete_sha256=t['complete.json']['sha256']),
        test=dict(identity_path=q['teacher-cache/identity.json']['path'],identity_sha256=q['teacher-cache/identity.json']['sha256'],
            labels_path=q['teacher-cache/labels.json']['path'],labels_sha256=q['teacher-cache/labels.json']['sha256'],
            raw_path=str(Path(q['teacher-cache/labels.json']['path']).parent/'rollout.pt'),
            raw_sha256=q['teacher-cache/labels.json']['summary']['raw_rollout_sha256'],
            raw_bytes=q['teacher-cache/labels.json']['summary']['raw_rollout_bytes'],
            selection_path=q['selection.json']['path'],selection_sha256=q['selection.json']['sha256']),
        canonical_train=dict(path=str(Path(RUNTIME_A800).parent/'data/train.parquet'),
            sha256=t['identity.json']['summary']['train_parquet_sha256']))
    payload=dict(source_root=args.source_root,code=base64.b64encode(raw).decode(),code_sha256=hashlib.sha256(raw).hexdigest(),
                 sources=sources,runtime=RUNTIME_A800,model=MODEL_A800,commit=args.commit)
    code=r'''
import base64,json,pathlib,hashlib,os,subprocess,time
p=PAYLOAD;b=pathlib.Path(p['source_root']);assert str(b).startswith('/data/shaoshitong/Search-G1-prefix-only-four-inputs-')
if b.exists():
 assert (b/'scan-claim.json').exists(), 'Occupied namespace; inspect only'
 print(json.dumps({'existing_claim':True,'base':str(b),'complete':(b/'inputs/scan-complete.json').exists(),'no_replay':True}));raise SystemExit
b.mkdir(mode=0o700)
def once(n,v):
 with (b/n).open('x') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
raw=base64.b64decode(p['code']);assert hashlib.sha256(raw).hexdigest()==p['code_sha256'];(b/'features.py').write_bytes(raw)
once('sources.json',p['sources']);once('scan-claim.json',dict(commit=p['commit'],feature_code_sha256=p['code_sha256'],actor_updates=0,head_updates=0,started_unix=time.time()))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=p['runtime'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
with (b/'scan.log').open('xb') as log:
 r=subprocess.run(['/data/search-g1/env/bin/python',str(b/'features.py'),'--scan-only','--sources',str(b/'sources.json'),'--output',str(b/'inputs'),'--model',p['model']],env=env,stdout=log,stderr=subprocess.STDOUT)
once('scan-exit.json',dict(exit_code=r.returncode,ended_unix=time.time()))
if r.returncode:
 print(json.dumps({'exit_code':r.returncode,'base':str(b),'error_tail':(b/'scan.log').read_text()[-1800:]}));raise SystemExit(r.returncode)
d=json.loads((b/'inputs/scan-complete.json').read_text())
print(json.dumps({'base':str(b),'exit_code':0,'count':d['count'],'encoder_global_max_tokens':d['encoder_global_max_tokens'],'head_max_tokens':d['head_max_tokens'],'max_policy_prefix_tokens':d.get('max_policy_prefix_tokens'),'max_raw_prefix_tokens':d.get('max_raw_prefix_tokens'),'representation':d.get('representation'),'prefix_window':d.get('prefix_window'),'prefix_source':d.get('prefix_source'),'prefix_coverage':d.get('prefix_coverage'),'segment_count':d.get('segment_count'),'scan_sha256':hashlib.sha256((b/'inputs/scan-complete.json').read_bytes()).hexdigest(),'sources_sha256':d['sources_sha256'],'input_bytes':sum(x['bytes'] for x in d['files'].values()),'teacher_regenerated':False,'GPU_used':False}))
'''.replace('PAYLOAD',repr(payload))
    r=subprocess.run(a800_route(code),capture_output=True,text=True,timeout=600)
    value=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,artifact_bytes_received_locally=False)
    record(args,'prefix-only-four-input-export',value)
    if r.returncode: raise RuntimeError('Source CPU export failed; inspect same source namespace')


def transfer_inputs(args):
    pub=subprocess.check_output(['ssh-keygen','-y','-f',str(Path.home()/'.ssh/id_ed25519_baai_a800')],text=True).strip()
    keys=subprocess.check_output(['ssh-keygen','-F','[ssh.platform-multi.baai.ac.cn]:2222'],text=True)
    keys='\n'.join(line for line in keys.splitlines() if not line.startswith('#'))
    payload=dict(source=args.source_root,target=args.target_inputs,public_key=pub,known_hosts=keys,a800_user=A800_USER)
    # This code runs on management. All input bytes flow A800 -> management ->
    # retained18 -> H100, never through this Mac stdout/stdin or filesystem.
    code=r'''
import json,subprocess,pathlib,shlex,os,tempfile
p=PAYLOAD;cfg=json.loads(pathlib.Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
assert os.environ.get('SSH_AUTH_SOCK'), 'Existing agent forwarding required; no credential copy'
with tempfile.TemporaryDirectory(prefix='prefix-only-four-transport-') as td:
 td=pathlib.Path(td);key=td/'public-key';known=td/'known-hosts';key.write_text(p['public_key']+'\n');known.write_text(p['known_hosts']+'\n')
 a=['ssh','-T','-i',str(key),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=5','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(known),'-o','HostKeyAlias=[ssh.platform-multi.baai.ac.cn]:2222','-p','2222',p['a800_user']+'@110.43.160.159']
 common=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ServerAliveInterval=5']
 h=common+['-o','ProxyCommand='+shlex.join(common+['-W','%h:%p',cfg['user']+'@'+cfg['gateway_ip']])]
 keys=cfg.get('trusted_target_host_keys',{}).get('h100-8')
 if keys:h+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(keys)])]
 h+=[cfg['user']+'@'+cfg['targets']['h100-8']]
 receiver="import sys,tarfile,pathlib,json,os; b=pathlib.Path("+repr(p['target'])+"); assert not b.exists(), 'Existing target inspect no replay'; b.mkdir(mode=0o700); t=tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz'); [t.extract(m,b) for m in t if not (pathlib.PurePosixPath(m.name).is_absolute() or '..' in pathlib.PurePosixPath(m.name).parts or m.issym() or m.islnk())]; print(json.dumps({'target':str(b),'transferred':True,'bytes':sum(q.stat().st_size for q in b.rglob('*') if q.is_file())}))"
 source_cmd=shlex.join(['tar','-czf','-','-C',p['source'],'inputs','sources.json','features.py','scan-claim.json','scan-exit.json'])
 with tempfile.TemporaryFile() as source_err:
  src=subprocess.Popen(a+[source_cmd],stdout=subprocess.PIPE,stderr=source_err)
  dst=subprocess.Popen(h+['python3 -c '+shlex.quote(receiver)],stdin=src.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE);src.stdout.close()
  out,err=dst.communicate(timeout=600);source_rc=src.wait(timeout=30);source_err.seek(0);source_error=source_err.read().decode()
 print(json.dumps({'source_exit':source_rc,'target_exit':dst.returncode,'target_result':json.loads(out) if out else None,'source_error':source_error,'target_error':err.decode(),'artifacts_via_Mac':False}))
 assert source_rc==dst.returncode==0
'''.replace('PAYLOAD',repr(payload))
    r=subprocess.run(['ssh','-A','-T','-o','ControlMaster=no','-o','ControlPath=none','mgmt-host',
                      'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=650)
    record(args,'prefix-only-four-input-remote-transfer',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,artifact_bytes_received_locally=False))
    if r.returncode: raise RuntimeError('Remote transfer incomplete; inspect same target without replay')


def deploy(args):
    archive_bytes=git('archive','--format=tar',args.commit)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as source:
        files={item.name:hashlib.sha256(source.extractfile(item).read()).hexdigest()
               for item in source if item.isfile()}
    archive=base64.b64encode(archive_bytes).decode()
    payload=dict(base=args.base,commit=args.commit,files=files,archive=archive,target_inputs=args.target_inputs,run_id=args.run_id,model=MODEL_H100,env=ENV_H100)
    code=r'''
import json,base64,io,tarfile,pathlib,hashlib,os,subprocess
p=PAYLOAD;b=pathlib.Path(p['base']);assert not b.exists(), 'Existing deployment inspect no replay';b.mkdir();(b/'preparation').mkdir();(b/'logs').mkdir();repo=b/'repo';repo.mkdir()
def sha(q):return hashlib.sha256(pathlib.Path(q).read_bytes()).hexdigest()
def once(q,v):
 with q.open('x') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
once(b/'preparation/deploy-intent.json',dict(commit=p['commit'],run_id=p['run_id']))
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['archive']))) as t:t.extractall(repo)
for name,digest in p['files'].items():assert sha(repo/name)==digest,name
stage=pathlib.Path(p['target_inputs']);proof=json.loads((stage/'inputs/scan-complete.json').read_text())
for name,v in proof['files'].items():assert sha(stage/'inputs'/name)==v['sha256']
sources=dict(model_revision=proof['encoder_revision'],model_files_sha256=proof['model_files_sha256'],prepared_scan=dict(path=str(stage/'inputs'),scan_complete_sha256=sha(stage/'inputs/scan-complete.json'),original_sources_sha256=proof['sources_sha256'],source_model_files_sha256=proof['model_files_sha256']))
once(b/'preparation/sources.json',sources)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
uuids=[v.split(',')[1].strip() for v in sorted(gpu.strip().splitlines(),key=lambda v:int(v.split(',')[0]))];assert len(uuids)==8
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
queue=subprocess.check_output(['squeue','-h','-w','h100-8','-t','RUNNING,COMPLETING','-o','%i'],text=True)
foreign_jobs=[];foreign_apps=[]
for job in queue.split():
 raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True);fields=dict(v.split('=',1) for v in raw.split() if '=' in v)
 assert fields['JobId']==job and fields['NodeList']=='h100-8'
 foreign_jobs.append(dict(job_id=job,user_id=fields['UserId'],start_time=fields['StartTime'],command=fields['Command']))
for line in apps.splitlines():
 if not line.strip():continue
 pid,gpu_uuid=[v.strip() for v in line.split(',')];proc=pathlib.Path('/proc')/pid
 try:
  raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split();identity=dict(pid=int(pid),start_ticks=int(f[19]),sid=int(f[3]),uid=proc.stat().st_uid,state=f[0])
 except FileNotFoundError:continue
 foreign_apps.append(dict(gpu_uuid=gpu_uuid,identity=identity))
c=dict(base=str(b),run_id=p['run_id'],source_commit=p['commit'],environment_root=p['env'],model_path=p['model'],sources_json=str(b/'preparation/sources.json'),sources_json_sha256=sha(b/'preparation/sources.json'),physical_gpu_uuids=uuids,master_ports=[29601,29602,29603,29604],preserved_foreign_allocations=foreign_jobs,preserved_foreign_gpu_processes=foreign_apps,prepared_inputs_source=str(stage/'inputs'))
once(b/'preparation/source-files.json',p['files']);once(b/'preparation/launch-config.json',c)
result=dict(base=str(b),commit=p['commit'],target_code_files=len(p['files']),all_target_code_sha_match=True,prepared_input_files_full_sha_match=True,prepared_count=proof['count'],encoder_global_max_tokens=proof['encoder_global_max_tokens'],head_max_tokens=proof['head_max_tokens'],max_policy_prefix_tokens=proof.get('max_policy_prefix_tokens'),max_raw_prefix_tokens=proof.get('max_raw_prefix_tokens'),representation=proof.get('representation'),prefix_window=proof.get('prefix_window'),prefix_source=proof.get('prefix_source'),prefix_coverage=proof.get('prefix_coverage'),segment_count=proof.get('segment_count'),no_new_rollout=True,submitted=False)
once(b/'preparation/deploy-receipt.json',result);print(json.dumps(result))
'''.replace('PAYLOAD',repr(payload))
    h100(args,code,'prefix-only-four-h100-code-deploy')


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['export','transfer','deploy','submit','observe'])
    for name in ('commit','source-root','target-inputs','base','run-id','reference-metadata','management-receipt','receipts'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if a.phase=='export':export_inputs(a)
    elif a.phase=='transfer':transfer_inputs(a)
    elif a.phase=='deploy':deploy(a)
    else:
        module='launch' if a.phase=='submit' else 'observe'
        argv=['submit','--base',a.base] if a.phase=='submit' else ['--base',a.base]
        code='import sys,json;sys.path.insert(0,'+repr(a.base+'/repo')+');sys.argv='+repr([module]+argv)+';from reproduction.prefix_only_correctness_ab.'+module+' import main;main()'
        h100(a,code,'prefix-only-four-'+a.phase)


if __name__=='__main__':main()
