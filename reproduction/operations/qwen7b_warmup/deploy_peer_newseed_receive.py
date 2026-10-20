"""Code-only immutable receiver deployment and guarded RAM-only launch.

A short-lived task-owned SSH ControlMaster carries code/credentials/small audit
summaries only. Archive payloads move HF->HKUST2->HKUST3->HPC3 independently.
"""
import argparse,contextlib,hashlib,importlib.util,json,os,shlex,stat,subprocess,tempfile,time
from pathlib import Path
from . import peer_newseed_receive as receiver

SNAPSHOT='newseed-20261021-185747-445fc691'
PREFLIGHT=r'''
import concurrent.futures,hashlib,json,requests,socket,sys,time
from urllib.parse import quote,urlsplit
secret=json.loads(sys.stdin.readline());record=secret['archive_probe']
assert record['archive_bytes']>1048576 and len(record['archive_sha256'])==64 and len(record['hf_archive_commit'])==40

def probe(route):
 label,port=route;started=time.monotonic();value=dict(exit=label,range_bytes_read=0,range_accepted=False)
 try:
  if port:
   sock=socket.create_connection(('127.0.0.1',port),timeout=1);sock.close()
  with requests.Session() as session:
   session.trust_env=False;session.headers['Authorization']='Bearer '+secret['hf_token']
   if port:session.proxies={'http':'http://127.0.0.1:'+str(port),'https':'http://127.0.0.1:'+str(port)}
   response=session.get('https://huggingface.co/api/datasets/Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020',timeout=(8,15));value['metadata_http_status']=response.status_code
   response.raise_for_status();info=response.json()
   assert info.get('private') is True and info.get('id','').lower()=='sutoonq/qwen25-7b-probe-h100-transfer-20261020'
   value['metadata_status']='PRIVATE_REPO_MATCH'
   response=session.post('https://huggingface.co/api/datasets/Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020/paths-info/'+record['hf_archive_commit'],json={'paths':[record['hf_archive_path']]},timeout=(8,15));response.raise_for_status();items=response.json()
   assert len(items)==1 and items[0]['path']==record['hf_archive_path'] and items[0]['size']==record['archive_bytes'] and items[0]['lfs']['oid']==record['archive_sha256']
   value['pinned_LFS_source_size_SHA_match']=True
   limit=1048576;h=hashlib.sha256();begin=time.monotonic()
   url='https://huggingface.co/datasets/Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020/resolve/'+record['hf_archive_commit']+'/'+quote(record['hf_archive_path'],safe='/')
   with session.get(url,headers={'Range':'bytes=0-'+str(limit-1)},stream=True,timeout=(8,15)) as response:
    value['range_http_status']=response.status_code;response.raise_for_status()
    if response.status_code!=206 or response.headers.get('Content-Range')!='bytes 0-%d/%d'%(limit-1,record['archive_bytes']):raise ValueError('Exact bounded range was not honored')
    for block in response.iter_content(65536):
     if not block:continue
     if value['range_bytes_read']+len(block)>limit:raise ValueError('Bounded range exceeded')
     value['range_bytes_read']+=len(block);h.update(block)
   assert value['range_bytes_read']==limit
   value.update(range_accepted=True,range_prefix_SHA256=h.hexdigest(),range_elapsed_seconds=time.monotonic()-begin,status='PRIVATE_API_AND_REAL_PINNED_ARCHIVE_RANGE_ACCEPTED_NOT_FULL_ARCHIVE_SHA')
 except Exception as error:
  value.update(status='FAILED',error_type=type(error).__name__)
  # Signed CDN query strings and exception messages must never be emitted.
  request=getattr(error,'request',None)
  if request is not None:value['failed_request_host_only']=urlsplit(request.url).hostname
 value['elapsed_seconds']=time.monotonic()-started;return value

routes=[('direct',None),('HKUST2-17894',17894),('HKUST2-17897',17897),('HKUST2-17898',17898)]
if secret.get('probe_port') is not None:
 assert secret['probe_port'] in (17894,17897,17898);routes=[('HKUST2-'+str(secret['probe_port']),secret['probe_port'])]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(probe,routes))
print(json.dumps(dict(status='READ_ONLY_OFFICIAL_HF_METADATA_AND_REAL_RANGE_PREFLIGHT',exits=out,proxy_environment_inherited=False,Mac_proxy_used=False,archive_bytes_read_only_in_remote_RAM=sum(r['range_bytes_read'] for r in out),full_archive_SHA_acceptance=False,epoch=time.time())))
'''
COMMON=r'''
import fcntl,hashlib,json,os,pathlib,subprocess,sys,time
b=pathlib.Path(inputs['base']);p=pathlib.Path(inputs['code']);assert b.parent==pathlib.Path('/data/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021') and b.name==inputs['snapshot']
assert p==b/'code'/inputs['code_sha256']/'peer_newseed_receive.py'
assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==inputs['code_sha256']
def atomic(p,v):
 q=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with q.open('w') as f:json.dump(v,f,sort_keys=True);f.write(chr(10));f.flush();os.fsync(f.fileno())
 q.replace(p)
def identity(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def live(value):return identity(value['pid'])==value['start_ticks']
'''
START=r'''
secret=json.loads(sys.stdin.readline());assert set(secret)=={'hf_token','hpc3_password'} and all(secret.values())
assert len(inputs['catalog_commit'])==40 and all(c in '0123456789abcdef' for c in inputs['catalog_commit'])
with (b/'launch.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (b/'launch.json').exists():
  v=json.loads((b/'launch.json').read_text());print(json.dumps(dict(status='EXISTING_RECEIVER_LAUNCH_NOT_REPLAYED',launch=v,actual_process_same_birth=live(v))))
 else:
  assert not (b/'launch-intent.json').exists(),'Unknown prior launch result; inspect before retry'
  if (b/'controller.json').exists():
   c=json.loads((b/'controller.json').read_text());cmd=pathlib.Path('/proc/%d/cmdline'%c['pid'])
   assert not cmd.exists() or str(p).encode() not in cmd.read_bytes(),'Existing receiver writer has no launcher receipt'
  intent=dict(status='RECEIVER_LAUNCH_INTENT_NOT_SUCCESS',code_sha256=inputs['code_sha256'],source_commit=inputs['source_commit'],catalog_commit=inputs['catalog_commit'],prefix=inputs['prefix'],hf_proxy=inputs['hf_proxy'],epoch=time.time())
  atomic(b/'launch-intent.json',intent)
  args=['python3','-u',str(p),'--action','receive','--base',str(b),'--snapshot',inputs['snapshot'],'--prefix',inputs['prefix'],'--catalog-commit',inputs['catalog_commit']]
  if inputs['hf_proxy']:args+=['--hf-proxy',inputs['hf_proxy']]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','SSHPASS'):env.pop(key,None)
  with (b/'receiver.log').open('ab') as log:
   child=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.DEVNULL,env=env,start_new_session=True)
   child.stdin.write(json.dumps(secret).encode()+bytes([10]));child.stdin.close()
  value=dict(intent,status='RECEIVER_STARTED_NOT_PROGRESS_VERIFIED',pid=child.pid,start_ticks=identity(child.pid),token_saved=False,password_saved=False,Mac_data_transit=False)
  atomic(b/'launch.json',value);print(json.dumps(value))
'''
STATUS=r'''
out=dict(base=str(b),observed_epoch=time.time(),code_sha256=inputs['code_sha256'],source_commit=inputs['source_commit'])
for name in ('source-publication.json','launch.json','controller.json','progress.json','retry.json','failure.json','complete.json'):
 q=b/name
 if q.exists():out[name]=json.loads(q.read_text())
if (b/'launch.json').exists():
 v=out['launch.json'];out['actual_process_same_birth']=live(v)
 if out['actual_process_same_birth']:
  cmd=pathlib.Path('/proc/%d/cmdline'%v['pid']).read_bytes();out['actual_owned_command']=str(p).encode() in cmd and str(b).encode() in cmd
  q=pathlib.Path('/proc/%d/io'%v['pid'])
  if q.exists():out['actual_io_counters']={k:int(v) for k,v in (s.split(':',1) for s in q.read_text().splitlines())}
out['actual_archive_files']={}
for q in (b/'archives').glob('*'):
 if q.is_file() and not q.is_symlink():out['actual_archive_files'][q.name]=q.stat().st_size
for name in ('download-accepted','target-accepted'):
 records=[json.loads(q.read_text()) for q in (b/name).glob('*.json')]
 out[name]=dict(groups=len(records),bytes=sum(r.get('archive_bytes',r.get('bytes',0)) for r in records))
print(json.dumps(out))
'''


def render(inputs,action):
    if action not in ('start','status'):raise ValueError('Unknown receiver control action')
    return 'inputs='+repr(inputs)+'\n'+COMMON+(START if action=='start' else STATUS)


def master_args(host,socket,proxy=None):
    # -M together with ControlMaster=yes changes OpenSSH to ASK mode.  The
    # noninteractive master then refuses sessions; never enable it twice.
    args=['ssh','-T','-S',socket,'-oControlMaster=yes','-oControlPersist=180','-fN',
        '-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8',
        '-oServerAliveInterval=15','-oServerAliveCountMax=3','-oProxyJump=none']
    if proxy:args+=['-oProxyCommand='+proxy]
    return args+[host]


def master_client(host,socket):
    # A lost/denied master must fail instead of silently opening another route.
    return ['ssh','-T','-S',socket,'-oControlMaster=no','-oBatchMode=yes',
        '-oStrictHostKeyChecking=yes','-oConnectTimeout=8','-oProxyJump=none',
        '-oProxyCommand=false',host]


@contextlib.contextmanager
def route(direct=False,physical=False,via_hkust1=False):
    """Never alter user SSH config or reuse/stop another task's master."""
    with tempfile.TemporaryDirectory(prefix='q7recv-',dir='/tmp') as folder:
        socket=str(Path(folder)/'ssh.sock');first='hkust1' if via_hkust1 else 'hkust2'
        args=master_args(first,socket)
        if direct:
            ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
            args[-1:-1]=['-B','en0','-b',ip,'-oProxyCommand=none']
        if physical:
            if direct:raise ValueError('Choose one explicit SSH socket binding')
            args[-1:-1]=['-oProxyCommand=/usr/bin/nc --apple-no-connectx -b en0 -G 8 %h %p']
        masters=[(socket,first)]
        try:
            result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=15)
            if result.returncode:raise RuntimeError('HKUST2 control authentication unavailable')
            if via_hkust1:
                hop=master_client('hkust1',socket);hop[-1:-1]=['-W','%h:%p']
                second=str(Path(folder)/'hkust2.sock');masters.append((second,'hkust2'))
                result=subprocess.run(master_args('hkust2',second,shlex.join(hop)),stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=15)
                if result.returncode:raise RuntimeError('HKUST2 strict authentication through HKUST1 unavailable')
                socket=second
            check=subprocess.run(['ssh','-S',socket,'-O','check','hkust2'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=8)
            if check.returncode:raise RuntimeError('Receiver control master is not actually available')
            yield master_client('hkust2',socket)
        finally:
            for own_socket,host in reversed(masters):
                if Path(own_socket).exists():subprocess.run(['ssh','-S',own_socket,'-O','exit',host],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)


def publication(repo):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(repo)).decode().strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=str(repo)).split()[0].decode()
    if commit!=remote:raise ValueError('Publish exact current receiver controls before using them')
    for path in (Path(receiver.__file__),Path(__file__)):
        if path.read_bytes()!=subprocess.check_output(['git','show',commit+':'+str(path.relative_to(repo))],cwd=str(repo)):
            raise ValueError('Unpublished receiver/control bytes')
    return commit


def token():
    p=Path.home()/'.cache/huggingface/token'
    if stat.S_IMODE(p.stat().st_mode)!=0o600:raise ValueError('HF token input file must have mode600')
    result=p.read_text().strip()
    if not result:raise ValueError('RAM HF token unavailable')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--action',choices=['stage','preflight','start','status'],required=True)
    parser.add_argument('--control-root',type=Path,required=True);parser.add_argument('--snapshot',default=SNAPSHOT);parser.add_argument('--catalog-commit');parser.add_argument('--hf-proxy');parser.add_argument('--direct-en0',action='store_true');parser.add_argument('--physical-en0',action='store_true',help='Darwin IP_BOUND_IF SSH socket only, no connectx or global route changes');parser.add_argument('--archive-binding',type=Path,help='Small actual source-plan/HF-receipt binding for CDN Range preflight')
    parser.add_argument('--via-hkust1',action='store_true',help='Task-owned HKUST1 TCP SSH control hop; existing local HKUST2 authentication, no payload transit');args=parser.parse_args()
    repo=Path(__file__).resolve().parents[3];commit=publication(repo);snapshot=receiver.snapshot_name(args.snapshot);base=receiver.K2/snapshot;digest=hashlib.sha256(Path(receiver.__file__).read_bytes()).hexdigest();code=base/'code'/digest/'peer_newseed_receive.py'
    inputs=dict(base=str(base),snapshot=snapshot,code=str(code),code_sha256=digest,source_commit=commit,prefix=receiver.PREFIX+snapshot,catalog_commit=args.catalog_commit,hf_proxy=args.hf_proxy)
    with route(args.direct_en0,args.physical_en0,args.via_hkust1) as ssh:
        if args.action=='stage':
            actual=receiver.install_code(ssh,base)
            assert actual==code
            script='import pathlib,hashlib,json,time; b=pathlib.Path(%r); p=pathlib.Path(%r); v=dict(status="RECEIVER_CODE_STAGED_ONLY",source_commit=%r,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,receiver_started=(b/"receiver.lock").exists(),catalog_created=(b/"catalog.json").exists(),epoch=time.time()); (b/"source-publication.json").write_text(json.dumps(v)); print(json.dumps(v))'%(str(base),str(code),commit)
            result=receiver.remote(ssh,'python3 -c '+shlex.quote(script),timeout=45)
        elif args.action=='preflight':
            if not args.archive_binding:raise ValueError('Actual source/HF first-archive binding required for real CDN preflight')
            archive=json.loads(args.archive_binding.read_bytes())['first_group']
            if (not archive['hf_archive_path'].startswith(receiver.PREFIX+snapshot+'/archives/') or
                    not all(c in '0123456789abcdef' for c in archive['hf_archive_commit']) or len(archive['hf_archive_commit'])!=40 or
                    not all(c in '0123456789abcdef' for c in archive['archive_sha256']) or len(archive['archive_sha256'])!=64):
                raise ValueError('Pinned authorized source/HF archive identity required')
            probe_port=None
            if args.hf_proxy:
                if args.hf_proxy not in ('http://127.0.0.1:17894','http://127.0.0.1:17897','http://127.0.0.1:17898'):raise ValueError('Only existing owned HKUST2 exits can be probed')
                probe_port=int(args.hf_proxy.rsplit(':',1)[1])
            result=receiver.remote(ssh,'python3 -c '+shlex.quote(PREFLIGHT),dict(hf_token=token(),archive_probe=archive,probe_port=probe_port),timeout=55)
        else:
            script=render(inputs,args.action)
            if len(('python3 -c '+shlex.quote(script)).encode())>=16000:raise ValueError('Receiver control command exceeds16KB')
            secret=None
            if args.action=='start':
                if not args.catalog_commit:raise ValueError('Pinned actual HF catalog commit required before start')
                p=args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py';spec=importlib.util.spec_from_file_location('receiver_private_control',str(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
                secret=dict(hf_token=token(),hpc3_password=m.password_for('mou176').decode())
            result=receiver.remote(ssh,'python3 -c '+shlex.quote(script),secret,timeout=55)
    out=args.control_root/'outputs/qwen7b-warmup-20261019'/('peer-newseed-receiver-'+args.action+'.json');out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
