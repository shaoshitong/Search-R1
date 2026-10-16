"""Code-only once deployment and native submission for the fresh group-joint experiment."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

REPO=Path(__file__).resolve().parents[2]


def archive_for_commit(commit,reuse_commit=None):
    def archived(revision):
        blob=subprocess.check_output(['git','-C',str(REPO),'archive','--format=tar.gz',revision])
        contents={};members={}
        with tarfile.open(fileobj=io.BytesIO(blob))as handle:
            for member in handle:
                if member.isdir():continue
                if not member.isfile():raise ValueError('Only regular source files may be deployed')
                contents[member.name]=handle.extractfile(member).read();members[member.name]=member
        return blob,contents,members
    blob,contents,members=archived(commit)
    files={name:hashlib.sha256(value).hexdigest()for name,value in contents.items()}
    reuse={}
    if reuse_commit:
        if len(reuse_commit)!=40 or any(ch not in'0123456789abcdef'for ch in reuse_commit):
            raise ValueError('Reuse source must be an exact committed source identity')
        _,old,_=archived(reuse_commit)
        reuse={name:digest for name,digest in files.items()
               if name in old and hashlib.sha256(old[name]).hexdigest()==digest}
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz')as handle:
            for name,data in contents.items():
                if name not in reuse:handle.addfile(members[name],io.BytesIO(data))
        blob=stream.getvalue()
    return dict(archive=blob,files=files,reuse_files=reuse,
                modes={name:member.mode for name,member in members.items()})


def deployment_script(base,commit,c,built,reuse_commit=None,reuse_repo=None):
    c=dict(c,base=base,source_commit=commit)
    payload=dict(base=base,commit=commit,config=c,files=built['files'],modes=built['modes'],
        archive=base64.b64encode(built['archive']).decode(),reuse_files=built['reuse_files'],
        reuse_source_commit=reuse_commit,reuse_source_repo=reuse_repo)
    return r'''
import base64,hashlib,io,json,os,pathlib,shutil,sys,tarfile,time
p=PAYLOAD;b=pathlib.Path(p['base']);c=p['config']
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def once(path,value):
 with path.open('x')as handle:json.dump(value,handle,sort_keys=True,allow_nan=False);handle.flush();os.fsync(handle.fileno())
assert str(b).startswith('/mnt/data/shaoshitong/search-g1-question20-groupjoint100-')and'..'not in b.parts
if b.exists():
 print(json.dumps(dict(status='EXISTING_DEPLOY_NAMESPACE_NO_REPLAY',intent=(b/'preparation/deploy-intent.json').exists(),receipt=read(b/'preparation/deploy-receipt.json')if(b/'preparation/deploy-receipt.json').exists()else None)));raise SystemExit
inventory=pathlib.Path(c['source_inventory_path']);test=pathlib.Path(c['fixed_test_features_path'])
assert str(inventory).startswith('/mnt/data/shaoshitong/')and'..'not in inventory.parts
assert str(test).startswith('/mnt/data/shaoshitong/')and'..'not in test.parts
assert sha(inventory)==c['source_inventory_sha256']and sha(test/'manifest.json')==c['fixed_test_manifest_sha256']
assert c['fresh_head']is True and not any(key in c for key in('resume_replay','source_checkpoint','source_adam_steps_per_rank'))
b.mkdir(mode=0o700);prep=b/'preparation';prep.mkdir();(b/'repo').mkdir();(b/'logs').mkdir()
once(prep/'deploy-intent.json',dict(run_id=c['run_id'],source_commit=p['commit'],file_count=len(p['files']),reuse_source_commit=p['reuse_source_commit'],reuse_source_repo=p['reuse_source_repo'],reuse_file_count=len(p['reuse_files']),created_unix=time.time()))
if p['reuse_files']:
 root=pathlib.Path(p['reuse_source_repo'])
 assert str(root).startswith('/mnt/data/shaoshitong/')and root.name=='repo'and'..'not in root.parts
 assert root.is_dir()and root.resolve(strict=True)==root
 for name,digest in p['reuse_files'].items():
  relative=pathlib.PurePosixPath(name);assert not relative.is_absolute()and'..'not in relative.parts
  source=root/name;assert source.is_file()and not source.is_symlink()and source.resolve(strict=True)==source
  assert root in source.parents and sha(source)==digest,name
  target=b/'repo'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);target.chmod(p['modes'][name])
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['archive'])))as handle:
 for member in handle:
  relative=pathlib.PurePosixPath(member.name);assert not relative.is_absolute()and'..'not in relative.parts
  if member.isdir():(b/'repo'/member.name).mkdir(parents=True,exist_ok=True)
  elif member.isfile():
   target=b/'repo'/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(handle.extractfile(member).read());target.chmod(member.mode)
for name,digest in p['files'].items():assert sha(b/'repo'/name)==digest,name
sys.path.insert(0,str(b/'repo'))
from reproduction.prefix20_group_joint100.launch import validate_config
validate_config(c,b)
shutil.copyfile(inventory,prep/'source-inventory.json')
assert sha(prep/'source-inventory.json')==c['source_inventory_sha256']
once(prep/'launch-config.json',c);once(prep/'source-files.json',p['files'])
result=dict(status='FRESH_GROUP_JOINT100_CODE_DEPLOYED_NOT_SUBMITTED',base=str(b),run_id=c['run_id'],source_commit=p['commit'],target_code_files=len(p['files']),all_target_code_full_sha_match=True,config_sha256=sha(prep/'launch-config.json'),source_inventory_sha256=c['source_inventory_sha256'],fixed_test_manifest_sha256=c['fixed_test_manifest_sha256'],fresh_head=True,old_head_adam_rng_imported=False,new_rollouts=0,training_encoding_mode=c['encoding_mode'],reused_code_files=len(p['reuse_files']),transferred_code_files=len(p['files'])-len(p['reuse_files']),code_archive_bytes=len(base64.b64decode(p['archive'])),large_payload_transferred_from_Mac=False)
once(prep/'deploy-receipt.json',result);print(json.dumps(result))
'''.replace('PAYLOAD',repr(payload))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('deploy','submit','query','observe'))
    parser.add_argument('--base',required=True);parser.add_argument('--commit');parser.add_argument('--config')
    parser.add_argument('--receipts',required=True);parser.add_argument('--reuse-source-commit');parser.add_argument('--reuse-source-repo')
    args=parser.parse_args()
    if bool(args.reuse_source_commit)!=bool(args.reuse_source_repo):parser.error('Both optional reuse-source fields are required together')
    if args.phase=='deploy':
        if not args.commit or not args.config:parser.error('Deploy requires a published commit and complete config')
        built=archive_for_commit(args.commit,args.reuse_source_commit)
        code=deployment_script(args.base,args.commit,json.loads(Path(args.config).read_text()),built,
            args.reuse_source_commit,args.reuse_source_repo)
    else:
        code='import sys,json;sys.path.insert(0,'+repr(args.base+'/repo')+');from reproduction.prefix20_group_joint100.launch import '+args.phase+';print(json.dumps('+args.phase+'('+repr(args.base)+'),allow_nan=False))'
    from reproduction.h1008_qprobe.management_transport import remote
    path=Path(args.receipts)/('question20-groupjoint100-'+args.phase+'-'+str(time.time_ns())+'.json')
    result=json.loads(remote(code,path,control_path=None,target='h100-8'))
    output=path.with_name(path.stem+'-result.json');output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),result=result),allow_nan=False))


if __name__=='__main__':main()
