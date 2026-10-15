"""Stage a new execution for the authenticated saved CP5; never resets official0."""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zlib
from reproduction.h1008_qprobe.management_transport import remote

REMOTE = '''import pathlib,json,hashlib,shutil,os,base64,zlib
P=pathlib.Path;meta=META
old=P('/mnt/data/shaoshitong/search-g1-qprobe3-stdloss-545d4cf-20261015-v2');base=P(meta['base'])
terminal=json.loads((old/'preparation/terminal3399-once-v1/RESULT.json').read_text())
assert terminal['job_id']==3399 and terminal['cleanup_true'] and terminal['cgroup_gone'] and terminal['original_processes_gone']
assert not base.exists(),'Existing deployment: inspect immutable intent/receipt; never replay'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
base.mkdir()
for d in ('data','preparation','logs','tmp','cache','artifacts','checkpoints'):(base/d).mkdir()
once(base/'preparation/deploy-intent.json',dict(source_commit=meta['commit'],source_job=3399,restore_step=5,reset_official0=False))
shutil.copytree(old/'repo',base/'repo',ignore=shutil.ignore_patterns('__pycache__'))
for name in meta['deleted']:(base/'repo'/name).unlink()
for name,text in json.loads(zlib.decompress(base64.b64decode(meta['changes']))).items():
 p=base/'repo'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
for name,digest in meta['files'].items():assert sha(base/'repo'/name)==digest,name
for d in ('env','bootstrap-tools','bootstrap-assets','assets','retriever-gpu-lib'):(base/d).symlink_to((old/d).resolve(),target_is_directory=True)
for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):(base/'data'/name).symlink_to((old/'data'/name).resolve())
config=json.loads((old/'preparation/h1008-launch-config.json').read_text())
config.update(role='restore_h1008_transformer3_question_probe200',base=str(base),source_commit=meta['commit'],
 restore_receipt=str(old/'artifacts/question-probe/completed-pairs/global_step_5/source-receipt.json'),
 restore_source_job_id=3399,restore_terminal_relative='terminal3399-once-v1/RESULT.json')
config['official_manifest']=str(base/'preparation/official-model-manifest-original.json')
(base/'preparation/official-model-manifest-original.json').write_bytes((old/'preparation/official-model-manifest-original.json').read_bytes())
(base/'question-features').symlink_to(P(config['precomputed_question_features']['path']),target_is_directory=True)
once(base/'preparation/source-files.json',meta['files']);once(base/'preparation/h1008-launch-config.json',config)
result=dict(base=str(base),source_commit=meta['commit'],verified_files=len(meta['files']),all_target_source_full_sha_match=True,restore_step=5,submitted=False)
once(base/'preparation/deploy-receipt.json',result);print(json.dumps(result))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--control-path',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()
    proof=json.loads(a.proof.read_text());assert proof['published_control_commit']==commit and proof['all_blobs_full_sha_match']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
    files=proof['files'];names=subprocess.check_output(['git','diff','--name-only','545d4cf',commit],cwd=repo).decode().splitlines()
    changes={n:(repo/n).read_text() for n in names if n in files};deleted=[n for n in names if n not in files]
    meta=dict(commit=commit,files=files,deleted=deleted,changes=base64.b64encode(zlib.compress(json.dumps(changes).encode())).decode(),
              base='/mnt/data/shaoshitong/search-g1-qprobe3-resume5-'+commit[:7]+'-20261015')
    packed=base64.b64encode(zlib.compress(json.dumps(meta).encode())).decode()
    expression="json.loads(zlib.decompress(base64.b64decode("+repr(packed)+")))"
    print(remote(REMOTE.replace('META',expression),a.evidence,a.control_path))


if __name__=='__main__':main()
