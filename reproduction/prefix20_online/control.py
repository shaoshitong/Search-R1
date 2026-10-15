"""Code-only once deployment and bounded submit/query/observe transport."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

REPO = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])


def target(args, code, role):
    from reproduction.h1008_qprobe.management_transport import remote
    socket = json.loads(Path(args.management_receipt).read_text())['socket']
    receipt = Path(args.receipts)/(role+'-'+str(time.time_ns())+'.json')
    output = remote(code, receipt, control_path=socket)
    print(json.dumps(dict(receipt=str(receipt), result=json.loads(output)), allow_nan=False))


def deploy(args):
    archive = git('archive', '--format=tar', args.commit)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        files = {entry.name:hashlib.sha256(source.extractfile(entry).read()).hexdigest()
                 for entry in source if entry.isfile()}
    config = json.loads(Path(args.config).read_text())
    config.update(base=args.base, source_commit=args.commit)
    payload = dict(base=args.base, commit=args.commit, files=files, config=config,
                   archive=base64.b64encode(archive).decode())
    code = r'''
import base64,hashlib,io,json,os,pathlib,tarfile,time
p=PAYLOAD;b=pathlib.Path(p['base'])
assert str(b).startswith('/mnt/data/shaoshitong/search-g1-prefix20-online1000-')
if b.exists():
 print(json.dumps(dict(status='EXISTING_NAMESPACE_INSPECT_NO_REPLAY',base=str(b),deploy_intent=(b/'preparation/deploy-intent.json').exists(),deploy_receipt=(b/'preparation/deploy-receipt.json').exists())));raise SystemExit
b.mkdir(mode=0o700);(b/'preparation').mkdir();(b/'logs').mkdir();(b/'repo').mkdir()
def once(path,value):
 with path.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
once(b/'preparation/deploy-intent.json',dict(commit=p['commit'],run_id=p['config']['run_id'],file_count=len(p['files']),created_unix=time.time()))
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['archive']))) as t:
 for member in t:
  name=pathlib.PurePosixPath(member.name)
  assert not name.is_absolute() and '..' not in name.parts
  if member.isdir():(b/'repo'/member.name).mkdir(parents=True,exist_ok=True)
  elif member.isfile():
   q=b/'repo'/member.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(member).read());q.chmod(member.mode)
for name,value in p['files'].items():assert sha(b/'repo'/name)==value,name
c=p['config'];cached=pathlib.Path(c['cached_features']);manifest=cached/'manifest.json'
actual=sha(manifest)
if c.get('cached_features_manifest_sha256'):assert actual==c['cached_features_manifest_sha256'],'Fixed cache binding differs'
c['cached_features_manifest_sha256']=actual
for key in ('environment_root','model_dir','cached_features','cached_inputs','canonical_train','retrieval_index','retrieval_corpus','retrieval_encoder','retriever_gpu_lib'):
 assert pathlib.Path(c[key]).is_absolute() and pathlib.Path(c[key]).exists(),key
assert (pathlib.Path(c['environment_root'])/'bin/python').is_file()
once(b/'preparation/launch-config.json',c);once(b/'preparation/source-files.json',p['files'])
proof=dict(status='CODE_DEPLOYED_NOT_SUBMITTED',base=str(b),commit=p['commit'],run_id=c['run_id'],target_code_files=len(p['files']),all_target_code_full_sha_match=True,cached_features_manifest_sha256=actual,config_sha256=sha(b/'preparation/launch-config.json'),models_or_rollouts_copied=False,checked_unix=time.time())
once(b/'preparation/deploy-receipt.json',proof);print(json.dumps(proof))
'''.replace('PAYLOAD', repr(payload))
    target(args, code, 'prefix20-online-deploy')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('deploy', 'submit', 'query', 'observe'))
    parser.add_argument('--base', required=True)
    parser.add_argument('--commit')
    parser.add_argument('--config')
    parser.add_argument('--management-receipt', required=True)
    parser.add_argument('--receipts', required=True)
    args = parser.parse_args()
    if args.phase == 'deploy':
        if not args.commit or not args.config:
            parser.error('Deploy requires commit and local small config JSON')
        deploy(args)
    else:
        function = 'observe' if args.phase == 'observe' else args.phase
        module = 'observe' if function == 'observe' else 'launch'
        code = ('import sys,json;sys.path.insert(0,'+repr(args.base+'/repo')+');'
                'from reproduction.prefix20_online.'+module+' import '+function+';'
                'print(json.dumps('+function+'('+repr(args.base)+'),allow_nan=False))')
        target(args, code, 'prefix20-online-'+args.phase)


if __name__ == '__main__':
    main()
