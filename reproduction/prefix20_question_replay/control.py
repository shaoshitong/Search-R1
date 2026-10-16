"""Published code-only deployment; every mutation has a distinct intent/receipt."""
import argparse,base64,hashlib,io,json,subprocess,tarfile,time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
REUSE_SOURCE_COMMIT='4bc683883a569d7feb9d5da87307612b872e8f3a'

def build_deploy_archive(commit,reuse_source_commit=None):
    """Transmit changed code only; remote reuse is independently SHA checked."""
    archive=subprocess.check_output(['git','-C',str(REPO),'archive','--format=tar.gz',commit])
    contents={};modes={};members={}
    with tarfile.open(fileobj=io.BytesIO(archive))as t:
        for item in t:
            if item.isdir():continue
            if not item.isfile():raise ValueError('Deployment archives contain regular code files only')
            contents[item.name]=t.extractfile(item).read();modes[item.name]=item.mode;members[item.name]=item
    files={name:hashlib.sha256(data).hexdigest()for name,data in contents.items()}
    reuse_files={}
    if reuse_source_commit is not None:
        if reuse_source_commit!=REUSE_SOURCE_COMMIT:raise ValueError('Reuse source must be the frozen original runtime')
        source=subprocess.check_output(['git','-C',str(REPO),'archive','--format=tar.gz',reuse_source_commit])
        with tarfile.open(fileobj=io.BytesIO(source))as t:
            source_files={item.name:hashlib.sha256(t.extractfile(item).read()).hexdigest()for item in t if item.isfile()}
        reuse_files={name:digest for name,digest in files.items()if source_files.get(name)==digest}
        delta=io.BytesIO()
        with tarfile.open(fileobj=delta,mode='w:gz')as t:
            for name,data in contents.items():
                if name not in reuse_files:t.addfile(members[name],io.BytesIO(data))
        archive=delta.getvalue()
    return dict(archive=archive,files=files,modes=modes,reuse_files=reuse_files,
                delta_file_count=len(files)-len(reuse_files))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('deploy','submit','query','observe'))
    p.add_argument('--base',required=True);p.add_argument('--commit');p.add_argument('--config');p.add_argument('--receipts',required=True)
    p.add_argument('--reuse-source-commit');p.add_argument('--reuse-source-repo')
    a=p.parse_args()
    if bool(a.reuse_source_commit)!=bool(a.reuse_source_repo):p.error('Both reuse-source options are required together')
    if a.phase=='deploy':
        built=build_deploy_archive(a.commit,a.reuse_source_commit)
        archive,files=built['archive'],built['files']
        c=json.loads(Path(a.config).read_text());c.update(base=a.base,source_commit=a.commit)
        payload=dict(base=a.base,config=c,commit=a.commit,files=files,modes=built['modes'],
                     archive=base64.b64encode(archive).decode(),reuse_files=built['reuse_files'],
                     reuse_source_commit=a.reuse_source_commit,reuse_source_repo=a.reuse_source_repo)
        code=r'''
import base64,hashlib,io,json,os,pathlib,shutil,sys,tarfile,time
p=PAYLOAD;b=pathlib.Path(p['base']);c=p['config']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(p,x):
 with p.open('x')as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
assert str(b).startswith('/mnt/data/shaoshitong/search-g1-question20-replay70-')
if b.exists():
 print(json.dumps(dict(status='EXISTING_DEPLOY_NAMESPACE_NO_REPLAY',intent=(b/'preparation/deploy-intent.json').exists(),receipt=read(b/'preparation/deploy-receipt.json')if(b/'preparation/deploy-receipt.json').exists()else None)));raise SystemExit
evidence=pathlib.Path(c['source_freeze_proof_path']);assert sha(evidence)==c['source_freeze_proof_sha256']
source=read(evidence)
assert source['source_epoch']==c['source_epoch']and source['source_checkpoint_sha256']==c['source_checkpoint_sha256']
assert source['owned_births_present']==0 and source['native_cgroup_exists']is False
assert source['actual_adam_steps_per_rank']==c['source_adam_steps_per_rank']
assert c['iterations']==70 and c['global_batch']==256 and c['learning_rate']==3e-6
b.mkdir(mode=0o700);prep=b/'preparation';prep.mkdir();(b/'repo').mkdir();(b/'logs').mkdir()
once(prep/'deploy-intent.json',dict(commit=p['commit'],run_id=c['run_id'],file_count=len(p['files']),reuse_source_commit=p['reuse_source_commit'],reuse_source_repo=p['reuse_source_repo'],reuse_file_count=len(p['reuse_files']),created_unix=time.time()))
if p['reuse_files']:
 root=pathlib.Path(p['reuse_source_repo'])
 assert p['reuse_source_commit']=='4bc683883a569d7feb9d5da87307612b872e8f3a'
 assert str(root).startswith('/mnt/data/shaoshitong/search-g1-prefix20-online1000-')and root.name=='repo'and'..'not in root.parts
 assert root.is_dir()and not root.is_symlink()and root.resolve(strict=True)==root
 for name,digest in p['reuse_files'].items():
  relative=pathlib.PurePosixPath(name);assert not relative.is_absolute()and'..'not in relative.parts
  src=root/name;assert src.is_file()and not src.is_symlink()and src.resolve(strict=True)==src
  assert root in src.parents and sha(src)==digest,name
  dest=b/'repo'/name;dest.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(src,dest);dest.chmod(p['modes'][name])
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['archive'])))as t:
 for x in t:
  path=pathlib.PurePosixPath(x.name);assert not path.is_absolute()and'..'not in path.parts
  if x.isdir():(b/'repo'/x.name).mkdir(parents=True,exist_ok=True)
  elif x.isfile():
   q=b/'repo'/x.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(x).read());q.chmod(x.mode)
for name,digest in p['files'].items():assert sha(b/'repo'/name)==digest,name
sources=[]
for item in source['replay_sources']:
 d=dict(split='train',epoch=item['source_data_epoch'])
 if item['kind']=='cached_inputs':d['inputs']=item['inputs']
 else:d.update(rollout_complete=item['rollout_complete'],sha256=item['sha256'])
 sources.append(d)
sources.append(dict(inputs=c['cached_inputs'],split='test'))
once(prep/'replay-sources.json',dict(sources=sources))
once(prep/'launch-config.json',c);once(prep/'source-files.json',p['files'])
result=dict(status='QUESTION20_REPLAY70_CODE_DEPLOYED_NOT_SUBMITTED',base=str(b),run_id=c['run_id'],source_commit=p['commit'],target_code_files=len(p['files']),all_target_code_full_sha_match=True,config_sha256=sha(prep/'launch-config.json'),source_epoch=c['source_epoch'],source_checkpoint_sha256=c['source_checkpoint_sha256'],source_adam_steps_per_rank=c['source_adam_steps_per_rank'],learning_rate=c['learning_rate'],new_rollouts=0,large_payload_transferred_from_Mac=False,reused_code_files=len(p['reuse_files']),transferred_code_files=len(p['files'])-len(p['reuse_files']),code_archive_bytes=len(base64.b64decode(p['archive'])))
once(prep/'deploy-receipt.json',result);print(json.dumps(result))
'''.replace('PAYLOAD',repr(payload))
    else:
        code='import sys,json;sys.path.insert(0,'+repr(a.base+'/repo')+');from reproduction.prefix20_question_replay.launch import '+a.phase+';print(json.dumps('+a.phase+'('+repr(a.base)+'),allow_nan=False))'
    from reproduction.h1008_qprobe.management_transport import remote
    path=Path(a.receipts)/('question20-replay70-'+a.phase+'-'+str(time.time_ns())+'.json')
    out=json.loads(remote(code,path,control_path=None,target='h100-8'))
    result=path.with_name(path.stem+'-result.json');result.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(path=str(result),sha256=hashlib.sha256(result.read_bytes()).hexdigest(),result=out),allow_nan=False))

if __name__=='__main__':main()
