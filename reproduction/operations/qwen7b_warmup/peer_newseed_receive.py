"""Remote-only HF -> HKUST2 -> HKUST3 -> HPC3 sealed raw snapshot receiver.

One stdlib file on all three receivers; requests is imported only on HKUST2.
Passwords/tokens arrive on stdin, remain in RAM/environment, never argv/logs.
No inference, sampling, model update, source deletion, or original bank writes.
"""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import time
from urllib.parse import quote, urlparse

REPO = 'Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
K2 = Path('/data/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021')
K3 = Path('/backup/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021')
TARGET = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-h100-20261021')
KNOWN = '/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts'
PREFIX = 'peer-newseed-to-hpc3-20261021/'
FORBIDDEN = ('.bin','.pt','.pth','.safetensors','.npy','.npz','.pkl','.pickle','.ckpt')
STATUS = 'PEER_NEWSEED_RAW_ALL_ARCHIVES_AND_FILES_FRESH_SHA256_VERIFIED'
MAX_JSON = 32*1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024**2), b''): h.update(data)
    return h.hexdigest()


def atomic(path, value):
    path=Path(path); secure_dir(path.parent)
    if path.is_symlink(): raise ValueError('Symlink metadata forbidden')
    tmp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    with tmp.open('wb') as stream:
        stream.write(canonical(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(tmp),str(path))


def immutable(path, value):
    path=Path(path)
    if path.exists():
        regular(path)
        if json.loads(path.read_bytes()) != value: raise ValueError('Immutable receiver binding differs')
    else: atomic(path,value)


def secure_dir(path):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts: raise ValueError('Absolute canonical directory required')
    current=Path(path.anchor)
    for part in path.parts[1:]:
        current=current/part
        if current.is_symlink(): raise ValueError('Symlink directory component forbidden')
        current.mkdir(exist_ok=True)
        if not current.is_dir(): raise ValueError('Expected receiver directory')
    return path


def regular(path):
    path=Path(path)
    current=Path(path.anchor)
    for part in path.parts[1:]:
        current=current/part
        if current.is_symlink(): raise ValueError('Symlink file component forbidden')
    if not stat.S_ISREG(path.lstat().st_mode): raise ValueError('Expected regular receiver file')
    return path


def safe_member(name):
    p=PurePosixPath(name)
    if (not isinstance(name,str) or not name or p.is_absolute() or str(p)!=name or
            any(x in p.parts for x in ('.','..')) or '\\' in name or
            any(c in name for c in ('\x00','\n','\r')) or p.suffix.lower() in FORBIDDEN):
        raise ValueError('Unsafe or non-raw archive member')
    return name


def group_name(value):
    if type(value) is int and value >= 0: return 'group-%05d'%value
    if not isinstance(value,str) or not re.fullmatch('[A-Za-z0-9_-]+',value):
        raise ValueError('Invalid archive group identity')
    return value


def snapshot_name(value):
    if not isinstance(value,str) or not re.fullmatch('[A-Za-z0-9_-]{8,80}',value):
        raise ValueError('Invalid frozen snapshot identity')
    return value


def validate_catalog(catalog, snapshot, prefix):
    snapshot_name(snapshot)
    if (catalog.get('status')!='ALL_PEER_NEWSEED_SOURCE_PRIVATE_HF_ARCHIVES_SHA_ACCEPTED_NOT_HPC3_COMPLETE' or
            catalog.get('repo')!=REPO or catalog.get('repo_type')!='dataset' or catalog.get('private') is not True or
            catalog.get('snapshot_id')!=snapshot or prefix!=PREFIX+snapshot or catalog.get('prefix')!=prefix or
            catalog.get('Mac_data_transit') is not False): raise ValueError('Pinned authorized private catalog required')
    source=catalog['source_snapshot']
    if (source.get('snapshot_id')!=snapshot or source.get('source_node')!='h100-10' or
            source.get('revision')!=REVISION or source.get('actor_updates')!=0 or
            not (source.get('fresh_all_source_files_SHA256') is True or
                 (source.get('all_copied_files_fresh_SHA_verified') is True and source.get('all_archives_fresh_SHA_and_member_SHA_verified') is True))):
        raise ValueError('Fresh raw official7B source acceptance required')
    if not source.get('groups') or len(source['groups'])!=len(catalog['groups']): raise ValueError('Archive count differs')
    names=set(); members=set(); source_by={group_name(g['group']):g for g in source['groups']}
    if len(source_by)!=len(source['groups']): raise ValueError('Duplicate source archive groups')
    for record in catalog['groups']:
        name=group_name(record['group']); original=source_by.get(name)
        if original is None or name in names: raise ValueError('Unknown or duplicate catalog archive')
        names.add(name)
        for key in ('group','archive','archive_bytes','archive_sha256','files','source_bytes','questions','trajectories'):
            if record.get(key)!=original.get(key): raise ValueError('Source/catalog archive fields differ')
        archive=PurePosixPath(record['archive'])
        if (archive.name in ('.','..') or not (archive.name.endswith('.tar') or archive.name.endswith('.tar.gz')) or
                not re.fullmatch('[A-Za-z0-9_.-]+',archive.name) or
                record.get('hf_archive_path')!=prefix+'/archives/'+archive.name or
                record.get('repo')!=REPO or record.get('repo_type')!='dataset' or record.get('private') is not True or
                record.get('source_revision')!=REVISION or record.get('snapshot_id')!=snapshot or
                record.get('status')!='PEER_SOURCE_PRIVATE_HF_ARCHIVE_SHA_ACCEPTED' or
                not re.fullmatch('[0-9a-f]{64}',record['archive_sha256']) or
                type(record['archive_bytes']) is not int or record['archive_bytes']<1): raise ValueError('Invalid accepted HF archive record')
        if type(record['questions']) is not int or record['questions']<0 or record['trajectories']!=5*record['questions']:
            raise ValueError('Invalid own G5 question count')
        own=set()
        for item in record['files']:
            name_=safe_member(item['path'])
            if name_ in members or name_ in own or type(item['bytes']) is not int or item['bytes']<0 or not re.fullmatch('[0-9a-f]{64}',item['sha256']):
                raise ValueError('Duplicate or invalid sealed member record')
            own.add(name_)
        members.update(own)
        if not own or sum(f['bytes'] for f in record['files'])!=record['source_bytes']:
            raise ValueError('Archive source/member size differs')
    if (source['files']!=len(members) or source['payload_bytes']!=sum(g['source_bytes'] for g in catalog['groups']) or
            source['questions']!=sum(g['questions'] for g in catalog['groups']) or
            source['trajectories']!=sum(g['trajectories'] for g in catalog['groups']) or source['trajectories']!=5*source['questions']):
        raise ValueError('Snapshot totals differ')
    return catalog


def verify_file(path, record):
    path=regular(path)
    if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
        raise ValueError('Fresh file SHA or size differs')


def archive_record(record):
    return dict(bytes=record['archive_bytes'],sha256=record['archive_sha256'])


def paths(base, record):
    archive=Path(base)/'archives'/Path(record['archive']).name
    return archive, archive.with_name(archive.name+'.part')


def safe_extract(archive, root, record):
    """Never extractall: exact unique regular members, bounded bytes, own partials."""
    root=secure_dir(root); expected={f['path']:f for f in record['files']}; seen=set()
    with tarfile.open(str(archive),'r|*') as tar:
        for member in tar:
            name=safe_member(member.name)
            if name in seen or name not in expected or not member.isfile() or member.issym() or member.islnk() or member.size!=expected[name]['bytes']:
                raise ValueError('Tar member type, size or exact membership differs')
            seen.add(name); dest=root/name; secure_dir(dest.parent)
            if dest.exists():
                verify_file(dest,expected[name]); continue
            tmp=dest.with_name(dest.name+'.extract-'+expected[name]['sha256']+'.part')
            if tmp.is_symlink(): raise ValueError('Symlink extraction partial forbidden')
            h=hashlib.sha256(); count=0
            with tar.extractfile(member) as inp, tmp.open('wb') as out:
                for data in iter(lambda:inp.read(1024**2),b''):
                    count+=len(data)
                    if count>member.size: raise ValueError('Tar member exceeds sealed size')
                    h.update(data); out.write(data)
                out.flush(); os.fsync(out.fileno())
            if count!=expected[name]['bytes'] or h.hexdigest()!=expected[name]['sha256']:
                raise ValueError('Extracted member SHA differs')
            os.chmod(str(tmp),0o444); os.replace(str(tmp),str(dest))
    if seen!=set(expected): raise ValueError('Tar members missing')
    # Independent second full read; an archive checksum is not file acceptance.
    for item in record['files']: verify_file(root/item['path'],item)
    return len(seen)


def ssh_k3():
    return ['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oUpdateHostKeys=no',
        '-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oServerAliveInterval=15',
        '-oServerAliveCountMax=3','shaoshitong@10.120.17.122']


def ssh_target():
    return ['sshpass','-e','ssh','-T','-oBatchMode=no','-oNumberOfPasswordPrompts=1',
        '-oStrictHostKeyChecking=yes','-oUpdateHostKeys=no','-oProxyCommand=none','-oProxyJump=none',
        '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile='+KNOWN,
        '-oConnectTimeout=8','-oServerAliveInterval=15','-oServerAliveCountMax=3','mou176@10.120.81.30']


def clean_env(password=None):
    env=dict(os.environ)
    for name in ('SSHPASS','HF_TOKEN','HUGGING_FACE_HUB_TOKEN'): env.pop(name,None)
    if password is not None:
        if not isinstance(password,str) or not password or '\n' in password: raise ValueError('RAM password required')
        env['SSHPASS']=password
    return env


def remote(ssh, command, payload=None, env=None, timeout=1800):
    data=canonical(payload)+b'\n' if payload is not None else None
    result=subprocess.run(ssh+[command],input=data,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
        env=env,timeout=timeout)
    if result.returncode: raise RuntimeError('Remote receiver stage failed')
    if len(result.stdout)>MAX_JSON: raise ValueError('Unexpected remote output size')
    return json.loads(result.stdout) if result.stdout.strip() else {}


def install_code(ssh, base, env=None):
    """Only code on stdin/control; resumable immutable 4096-byte SHA chunks."""
    raw=Path(__file__).read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    root=Path(base)/'code'/digest; filename=root/'peer_newseed_receive.py'
    for index,start in enumerate(range(0,len(raw),4096)):
        block=raw[start:start+4096]
        code=('import base64,hashlib,pathlib; p=pathlib.Path(%r); '
            'assert not any(q.is_symlink() for q in (p,)+tuple(p.parents)); p.mkdir(parents=True,exist_ok=True); '
            'b=base64.b64decode(%r); f=p/%r; '
            'assert not f.is_symlink(); assert not f.exists() or f.read_bytes()==b; '
            'f.write_bytes(b) if not f.exists() else None')%(str(root),base64.b64encode(block).decode(),'%05d.chunk'%index)
        if len(('python3 -c '+shlex.quote(code)).encode())>=16000: raise ValueError('Oversized code control command')
        remote(ssh,'python3 -c '+shlex.quote(code),env=env,timeout=60)
    code=('import hashlib,pathlib; p=pathlib.Path(%r); b=b"".join((p/("%%05d.chunk"%%i)).read_bytes() for i in range(%d)); '
        'assert hashlib.sha256(b).hexdigest()==%r; f=p/"peer_newseed_receive.py"; '
        'assert not f.is_symlink(); assert not f.exists() or f.read_bytes()==b; '
        'f.write_bytes(b) if not f.exists() else None')%(str(root),(len(raw)+4095)//4096,digest)
    remote(ssh,'python3 -c '+shlex.quote(code),env=env,timeout=60)
    return filename


def invoke(ssh, code, action, base, payload, env=None):
    command=shlex.join(['python3',str(code),'--action',action,'--base',str(base)])
    return remote(ssh,command,payload,env=env)


def remote_has_final(ssh, path, size, env=None):
    code='import json,pathlib; p=pathlib.Path(%r); print(json.dumps({"present":p.is_file() and not p.is_symlink() and p.stat().st_size==%d}))'%(str(path),size)
    return remote(ssh,'python3 -c '+shlex.quote(code),env=env,timeout=45)['present']


def transfer(ssh, source, dest, lock, base, stage, env=None):
    # --rsync-path holds an actual remote flock during the only archive writer.
    transport=shlex.join(ssh[:-1]); command=['rsync','--partial','--append-verify','--protect-args','--no-owner','--no-group',
        '--rsync-path='+shlex.join(['flock','-n',str(lock),'rsync']),'-e',transport,str(source),
        ssh[-1]+':'+str(dest)]
    with subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env) as process:
        started=time.time()
        while process.poll() is None:
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                code='import json,pathlib; p=pathlib.Path(%r); print(json.dumps({"bytes":p.stat().st_size if p.exists() else 0}))'%str(dest)
                actual=remote(ssh,'python3 -c '+shlex.quote(code),env=env,timeout=30)
                atomic(Path(base)/'progress.json',dict(stage=stage,bytes=actual['bytes'],total=Path(source).stat().st_size,epoch=time.time(),Mac_data_transit=False))
            if time.time()-started>1800:
                process.terminate(); process.wait(timeout=10); raise TimeoutError('Remote archive transfer timeout')
        if process.returncode: raise RuntimeError('Remote rsync failed')


def setup(base, parent, catalog, binding):
    base=Path(base)
    if base.parent!=parent or base.name!=catalog['snapshot_id']: raise ValueError('Independent approved snapshot root required')
    validate_catalog(catalog,base.name,binding['prefix'])
    if not re.fullmatch('[0-9a-f]{40}',binding['catalog_commit']) or not re.fullmatch('[0-9a-f]{64}',binding['catalog_sha256']):
        raise ValueError('Pinned catalog commit/SHA required')
    if hashlib.sha256(binding['catalog_bytes'].encode()).hexdigest()!=binding['catalog_sha256'] or json.loads(binding['catalog_bytes'])!=catalog:
        raise ValueError('Original catalog bytes differ')
    secure_dir(base); secure_dir(base/'archives'); secure_dir(base/'locks')
    immutable(base/'catalog.json',catalog); immutable(base/'catalog-binding.json',binding)
    return base


def load(base):
    base=Path(base); return json.loads((base/'catalog.json').read_bytes()),json.loads((base/'catalog-binding.json').read_bytes())


def pick(catalog,name):
    found=[g for g in catalog['groups'] if group_name(g['group'])==name]
    if len(found)!=1: raise ValueError('Unknown accepted archive group')
    return found[0]


def accepted_archive(base,record):
    final,part=paths(base,record)
    if final.exists(): verify_file(final,archive_record(record))
    else:
        verify_file(part,archive_record(record)); os.replace(str(part),str(final))
    return final


def target_init(base, payload):
    setup(base,TARGET,payload['catalog'],payload['binding'])
    return dict(status='INDEPENDENT_TARGET_PLAN_BOUND',Mac_data_transit=False)


def target_accept(base,payload):
    catalog,binding=load(base); setup(base,TARGET,catalog,binding); name=group_name(payload['group']); record=pick(catalog,name)
    with (base/'locks'/(name+'.lock')).open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        archive=accepted_archive(base,record); count=safe_extract(archive,base/'raw',record)
        result=dict(status='TARGET_ARCHIVE_AND_ALL_MEMBER_FILES_FRESH_SHA_VERIFIED',group=name,
            archive_bytes=record['archive_bytes'],archive_sha256=record['archive_sha256'],files=count,
            catalog_sha256=binding['catalog_sha256'],Mac_data_transit=False,epoch=time.time())
        atomic(base/'accepted'/(name+'.json'),result); return result


def target_finalize(base,payload):
    catalog,binding=load(base); setup(base,TARGET,catalog,binding)
    with (base/'locks/final.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        count=0
        for record in catalog['groups']:
            name=group_name(record['group']); accepted=json.loads(regular(base/'accepted'/(name+'.json')).read_bytes())
            if accepted['catalog_sha256']!=binding['catalog_sha256']: raise ValueError('Acceptance binding differs')
            verify_file(paths(base,record)[0],archive_record(record))
            for item in record['files']: verify_file(base/'raw'/item['path'],item); count+=1
        result=dict(status=STATUS,snapshot_id=base.name,target=str(base/'raw'),catalog_commit=binding['catalog_commit'],
            catalog_sha256=binding['catalog_sha256'],groups=len(catalog['groups']),files=count,
            archive_bytes=sum(g['archive_bytes'] for g in catalog['groups']),questions=catalog['source_snapshot']['questions'],
            trajectories=catalog['source_snapshot']['trajectories'],all_archive_SHA256=True,fresh_all_file_SHA256=True,
            source_all_inputs_sealed_at_freeze=catalog['source_snapshot'].get('source_all_inputs_sealed_at_freeze'),
            feature_bank_created=False,actor_updates=0,probe_updates=0,Mac_data_transit=False,epoch=time.time())
        atomic(base/'complete.json',result); return result


def h3_init(base,payload):
    base=setup(base,K3,payload['catalog'],payload['binding']); env=clean_env(payload['hpc3_password'])
    code=install_code(ssh_target(),TARGET/base.name,env)
    result=invoke(ssh_target(),code,'target-init',TARGET/base.name,dict(catalog=payload['catalog'],binding=payload['binding']),env)
    immutable(base/'target-code.json',dict(path=str(code),sha256=sha(__file__)))
    return result


def h3_forward(base,payload):
    catalog,binding=load(base); base=setup(base,K3,catalog,binding); name=group_name(payload['group']); record=pick(catalog,name)
    env=clean_env(payload['hpc3_password']); code=Path(json.loads((base/'target-code.json').read_bytes())['path']); target=TARGET/base.name
    with (base/'locks'/(name+'.lock')).open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        archive=accepted_archive(base,record)
        atomic(base/'accepted'/(name+'.json'),dict(status='HKUST3_ARCHIVE_FRESH_SHA_VERIFIED',group=name,
            bytes=record['archive_bytes'],sha256=record['archive_sha256'],epoch=time.time()))
        target_final,part=paths(target,record)
        if not remote_has_final(ssh_target(),target_final,record['archive_bytes'],env):
            transfer(ssh_target(),archive,part,target/'locks'/(name+'.lock'),base,'HKUST3_TO_HPC3',env)
        result=invoke(ssh_target(),code,'target-accept',target,dict(group=name),env)
        atomic(base/'target-accepted'/(name+'.json'),result); return result


def h3_finalize(base,payload):
    catalog,binding=load(base); setup(base,K3,catalog,binding); env=clean_env(payload['hpc3_password'])
    code=Path(json.loads((base/'target-code.json').read_bytes())['path'])
    result=invoke(ssh_target(),code,'target-finalize',TARGET/base.name,{},env)
    atomic(base/'complete.json',result); return result


def api_json(session,method,url,**kwargs):
    response=session.request(method,url,timeout=(10,45),**kwargs); response.raise_for_status()
    if len(response.content)>MAX_JSON: raise ValueError('HF metadata response too large')
    return response.json()


def verify_inventory(catalog,inventory):
    if not isinstance(inventory,list): raise ValueError('HF inventory must list actual archive objects')
    by={x['path']:x for x in inventory}
    expected={g['hf_archive_path'] for g in catalog['groups']}
    if len(by)!=len(inventory) or set(by)!=expected: raise ValueError('Pinned HF archive inventory differs')
    for record in catalog['groups']:
        item=by[record['hf_archive_path']]; lfs=item.get('lfs',{})
        if item.get('size')!=record['archive_bytes'] or lfs.get('oid',lfs.get('sha256'))!=record['archive_sha256']:
            raise ValueError('Source/HF LFS archive SHA or size differs')


def download(session,base,record,commit,retries=50):
    final,part=paths(base,record)
    if final.exists(): verify_file(final,archive_record(record)); return final
    if part.exists(): regular(part)
    url='https://huggingface.co/datasets/'+REPO+'/resolve/'+commit+'/'+quote(record['hf_archive_path'],safe='/')
    for attempt in range(1,retries+1):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if offset>record['archive_bytes']: raise ValueError('Retained partial exceeds sealed archive size')
            if offset<record['archive_bytes']:
                headers={'Range':'bytes=%d-'%offset} if offset else {}
                with session.get(url,headers=headers,stream=True,timeout=(10,30)) as response:
                    response.raise_for_status()
                    if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith('bytes %d-'%offset)):
                        raise ValueError('HF ignored exact resume range; retained partial preserved')
                    if not offset and response.status_code!=200: raise ValueError('Unexpected full archive HTTP status')
                    last=0
                    with part.open('ab') as out:
                        for data in response.iter_content(1024**2):
                            if not data: continue
                            if offset+len(data)>record['archive_bytes']: raise ValueError('HF stream exceeds sealed archive size')
                            out.write(data); offset+=len(data)
                            if time.time()-last>=8:
                                out.flush(); atomic(base/'progress.json',dict(stage='HF_TO_HKUST2',group=group_name(record['group']),
                                    bytes=offset,total=record['archive_bytes'],epoch=time.time(),Mac_data_transit=False)); last=time.time()
                        out.flush(); os.fsync(out.fileno())
            verify_file(part,archive_record(record)); os.replace(str(part),str(final))
            atomic(base/'download-accepted'/(group_name(record['group'])+'.json'),dict(status='SOURCE_HF_HKUST2_ARCHIVE_SHA_VERIFIED',
                bytes=record['archive_bytes'],sha256=record['archive_sha256'],epoch=time.time())); return final
        except ValueError: raise
        except Exception as error:
            atomic(base/'retry.json',dict(stage='HF_TO_HKUST2',group=group_name(record['group']),attempt=attempt,
                error_type=type(error).__name__,epoch=time.time()))
            if attempt==retries: raise
            time.sleep(min(20,2*attempt))


def receive(base,snapshot,prefix,commit,secret,hf_proxy=None):
    if sys.platform=='darwin' or base.parent!=K2 or base.name!=snapshot: raise ValueError('Run payload receiver only on approved HKUST2 remote storage')
    if not re.fullmatch('[0-9a-f]{40}',commit) or prefix!=PREFIX+snapshot_name(snapshot): raise ValueError('Exact pinned catalog commit/prefix required')
    secure_dir(base)
    with (base/'receiver.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable(base/'receive-intent.json',dict(snapshot_id=snapshot,prefix=prefix,catalog_commit=commit,receiver_sha256=sha(__file__),Mac_data_transit=False))
        import requests
        token=secret['hf_token']; password=secret['hpc3_password']
        if not token or not password: raise ValueError('RAM credentials required')
        with requests.Session() as session:
            # No inherited historic Mac reverse-proxy/environment configuration.
            session.trust_env=False
            if hf_proxy:
                proxy=urlparse(hf_proxy)
                if (proxy.scheme!='http' or proxy.hostname not in ('127.0.0.1','localhost') or
                        proxy.port not in (17894,17897,17898) or proxy.username or proxy.password or
                        proxy.path not in ('','/') or proxy.query or proxy.fragment):
                    raise ValueError('Only explicit owned HKUST2 local HF exits are allowed')
                session.proxies={'http':hf_proxy,'https':hf_proxy}
            session.headers['Authorization']='Bearer '+token
            info=api_json(session,'GET','https://huggingface.co/api/datasets/'+REPO)
            if info.get('id','').lower()!=REPO.lower() or info.get('private') is not True: raise ValueError('Authorized private dataset required')
            url='https://huggingface.co/datasets/'+REPO+'/resolve/'+commit+'/'+prefix+'/catalog.json'
            response=session.get(url,timeout=(10,45)); response.raise_for_status()
            if len(response.content)>MAX_JSON: raise ValueError('Catalog exceeds small metadata limit')
            raw=response.content; catalog=validate_catalog(response.json(),snapshot,prefix)
            binding=dict(prefix=prefix,catalog_commit=commit,catalog_path=prefix+'/catalog.json',
                catalog_bytes=raw.decode('utf-8'),catalog_sha256=hashlib.sha256(raw).hexdigest())
            setup(base,K2,catalog,binding)
            inventory=api_json(session,'POST','https://huggingface.co/api/datasets/'+REPO+'/paths-info/'+commit,
                json={'paths':[g['hf_archive_path'] for g in catalog['groups']]})
            verify_inventory(catalog,inventory); immutable(base/'hf-pinned-inventory.json',inventory)
            atomic(base/'controller.json',dict(pid=os.getpid(),snapshot_id=snapshot,catalog_commit=commit,
                stage='SOURCE_HF_PINNED_CATALOG_ACCEPTED',token_saved=False,password_saved=False,
                HF_exit='HKUST2_owned_local_proxy' if hf_proxy else 'HKUST2_direct',Mac_data_transit=False,epoch=time.time()))
            h3base=K3/snapshot; code=install_code(ssh_k3(),h3base)
            invoke(ssh_k3(),code,'h3-init',h3base,dict(catalog=catalog,binding=binding,hpc3_password=password),clean_env())
            for record in catalog['groups']:
                name=group_name(record['group']); archive=download(session,base,record,commit)
                h3_final,part=paths(h3base,record)
                if not remote_has_final(ssh_k3(),h3_final,record['archive_bytes'],clean_env()):
                    transfer(ssh_k3(),archive,part,h3base/'locks'/(name+'.lock'),base,'HKUST2_TO_HKUST3',clean_env())
                result=invoke(ssh_k3(),code,'h3-forward',h3base,dict(group=name,hpc3_password=password),clean_env())
                atomic(base/'target-accepted'/(name+'.json'),result)
                print(json.dumps(dict(group=name,status=result['status'],archive_bytes=record['archive_bytes'])),flush=True)
            result=invoke(ssh_k3(),code,'h3-finalize',h3base,dict(hpc3_password=password),clean_env())
            atomic(base/'complete.json',result); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['receive','h3-init','h3-forward','h3-finalize','target-init','target-accept','target-finalize'],required=True)
    parser.add_argument('--base',type=Path,required=True); parser.add_argument('--snapshot'); parser.add_argument('--prefix'); parser.add_argument('--catalog-commit'); parser.add_argument('--hf-proxy',help='Optional owned HKUST2 HTTP exit on localhost:17894/17897/17898; no inherited proxies')
    args=parser.parse_args()
    try:
        if sys.platform=='darwin': raise ValueError('Remote receiver cannot run payload actions on Mac')
        raw=sys.stdin.buffer.readline(MAX_JSON+1)
        if len(raw)>MAX_JSON: raise ValueError('Input metadata too large')
        payload=json.loads(raw)
        if args.action=='receive': result=receive(args.base,args.snapshot,args.prefix,args.catalog_commit,payload,args.hf_proxy)
        else:
            if args.base.parent not in (K3,TARGET): raise ValueError('Independent approved remote target required')
            function={'h3-init':h3_init,'h3-forward':h3_forward,'h3-finalize':h3_finalize,
                'target-init':target_init,'target-accept':target_accept,'target-finalize':target_finalize}[args.action]
            result=function(args.base,payload)
        print(json.dumps(result,sort_keys=True),flush=True)
    except BaseException as error:
        # Never write exception text: requests/SSH exceptions may contain URLs.
        if args.base.parent in (K2,K3,TARGET) and args.base.exists():
            atomic(args.base/'failure.json',dict(stage=args.action,error_type=type(error).__name__,epoch=time.time()))
        raise SystemExit(1)


if __name__=='__main__': main()
