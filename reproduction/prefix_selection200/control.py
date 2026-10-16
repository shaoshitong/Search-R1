"""Code-only deployment and one native submission for prefix-selection200."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

from reproduction.prefix20_group_joint100.control import archive_for_commit


def deployment_script(base, commit, config, built, reuse_commit=None, reuse_repo=None):
    payload = dict(base=base, commit=commit, config=dict(config, base=base, source_commit=commit),
        files=built['files'], modes=built['modes'], archive=base64.b64encode(built['archive']).decode(),
        reuse_files=built['reuse_files'], reuse_commit=reuse_commit, reuse_repo=reuse_repo)
    return r'''
import base64,hashlib,io,json,os,pathlib,shutil,sys,tarfile,time
p=PAYLOAD;b=pathlib.Path(p['base']);c=p['config']
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def once(path,value):
 with path.open('x')as handle:json.dump(value,handle,sort_keys=True,allow_nan=False);handle.flush();os.fsync(handle.fileno())
assert str(b).startswith('/mnt/data/shaoshitong/search-g1-prefix-selection200-')and'..'not in b.parts
if b.exists():
 print(json.dumps(dict(status='EXISTING_DEPLOY_NAMESPACE_NO_REPLAY',intent=(b/'preparation/deploy-intent.json').exists(),receipt=read(b/'preparation/deploy-receipt.json')if(b/'preparation/deploy-receipt.json').exists()else None)));raise SystemExit
# Reuse already independently validated source weights. Read the small actual
# CPU proof plus exact source stat; do not repeat an extra GPU forward gate.
proof_path=pathlib.Path(c['probe_source_cpu_proof']);assert sha(proof_path)==c['probe_source_cpu_proof_sha256'];proof=read(proof_path)
source=pathlib.Path(c['probe_source_checkpoint']);stat=source.stat()
assert stat.st_size==c['probe_source_bytes'] and stat.st_ino==c['probe_source_inode']
assert c['probe_source_sha256'] in json.dumps(proof)
assert sha(pathlib.Path(c['fixed_test_features'])/'manifest.json')==c['fixed_test_manifest_sha256']
assert sha(pathlib.Path(c['canonical_train']))==c['canonical_train_sha256']
assert sha(pathlib.Path(c['grounding_manifest']))==c['grounding_manifest_sha256']
for key in ('validation_questions','final_questions'):
 assert sha(pathlib.Path(c[key]))==c[key+'_sha256']
b.mkdir(mode=0o700);prep=b/'preparation';prep.mkdir();(b/'repo').mkdir();(b/'logs').mkdir()
once(prep/'deploy-intent.json',dict(run_id=c['run_id'],source_commit=p['commit'],file_count=len(p['files']),reuse_commit=p['reuse_commit'],created_unix=time.time()))
if p['reuse_files']:
 root=pathlib.Path(p['reuse_repo']);assert root.resolve(strict=True)==root and root.name=='repo'
 for name,digest in p['reuse_files'].items():
  rel=pathlib.PurePosixPath(name);assert not rel.is_absolute()and'..'not in rel.parts
  source=root/name;assert source.is_file()and not source.is_symlink()and sha(source)==digest,name
  target=b/'repo'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);target.chmod(p['modes'][name])
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['archive'])))as handle:
 for member in handle:
  rel=pathlib.PurePosixPath(member.name);assert not rel.is_absolute()and'..'not in rel.parts
  if member.isdir():(b/'repo'/member.name).mkdir(parents=True,exist_ok=True)
  elif member.isfile():
   target=b/'repo'/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(handle.extractfile(member).read());target.chmod(member.mode)
  else:raise ValueError('Code archive must contain regular files only')
for name,digest in p['files'].items():assert sha(b/'repo'/name)==digest,name
sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_selection200.launch import validate_config
validate_config(c,b)
once(prep/'launch-config.json',c);once(prep/'source-files.json',p['files'])
result=dict(status='PREFIX_SELECTION200_CODE_DEPLOYED_NOT_SUBMITTED',base=str(b),run_id=c['run_id'],source_commit=p['commit'],target_code_files=len(p['files']),all_target_code_full_sha_match=True,config_sha256=sha(prep/'launch-config.json'),reused_code_files=len(p['reuse_files']),code_archive_bytes=len(base64.b64decode(p['archive'])),large_payload_transferred_from_Mac=False)
once(prep/'deploy-receipt.json',result);print(json.dumps(result))
'''.replace('PAYLOAD', repr(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('deploy', 'submit', 'query', 'observe'))
    parser.add_argument('--base', required=True); parser.add_argument('--commit'); parser.add_argument('--config')
    parser.add_argument('--receipts', required=True); parser.add_argument('--reuse-source-commit'); parser.add_argument('--reuse-source-repo')
    args = parser.parse_args()
    if bool(args.reuse_source_commit) != bool(args.reuse_source_repo):
        parser.error('Both source reuse fields are required together')
    if args.phase == 'deploy':
        if not args.commit or not args.config: parser.error('Deploy requires a published commit/config')
        built = archive_for_commit(args.commit, args.reuse_source_commit)
        code = deployment_script(args.base, args.commit, json.loads(Path(args.config).read_text()), built,
                                 args.reuse_source_commit, args.reuse_source_repo)
    else:
        code = 'import sys,json;sys.path.insert(0,'+repr(args.base+'/repo')+');from reproduction.prefix_selection200.launch import '+args.phase+';print(json.dumps('+args.phase+'('+repr(args.base)+'),allow_nan=False))'
    from reproduction.h1008_qprobe.management_transport import remote
    path = Path(args.receipts) / ('prefix-selection200-'+args.phase+'-'+str(time.time_ns())+'.json')
    result = json.loads(remote(code, path, control_path=None, target='h100-8'))
    output = path.with_name(path.stem+'-result.json'); output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(), result=result), allow_nan=False))


if __name__ == '__main__': main()
