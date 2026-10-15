"""Stage new std-loss code only; leave the running predecessor untouched."""
import datetime
import gzip
import json
from pathlib import Path
import shlex
import subprocess
import sys
import base64
import concurrent.futures
import hashlib
import uuid

REMOTE = '''import sys,json,pathlib,io,tarfile,hashlib,os
P=pathlib.Path;stream=sys.stdin.buffer;n=int.from_bytes(stream.read(8),'big');meta=json.loads(stream.read(n));archive=stream.read(meta['archive_bytes'])
assert len(archive)==meta['archive_bytes']
base=P(meta['base']);old=P('/mnt/data/shaoshitong/search-g1-qprobe3-h1008-003a0b0-20261015-v3')
assert not base.exists(),'Deployment claimed; inspect intent/receipt, never replay'
run=json.loads((old/'logs/run-manifest.json').read_text());assert run['run_id']=='4e2fd299bf5e4f3e86d910661f6389b9'
assert run['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
def once(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
base.mkdir()
for d in ('repo','data','preparation','logs','tmp','cache','artifacts','checkpoints'):(base/d).mkdir()
once(base/'preparation/deploy-intent.json',dict(commit=meta['commit'],prior_job=3397,live_runtime_changed=False))
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
 for m in t.getmembers():
  p=P(m.name);assert not p.is_absolute() and '..' not in p.parts and (m.isfile() or m.isdir())
 t.extractall(base/'repo')
for name,sha in meta['files'].items():assert hashlib.sha256((base/'repo'/name).read_bytes()).hexdigest()==sha,name
for d in ('env','bootstrap-tools','bootstrap-assets','assets','retriever-gpu-lib'):
 assert (old/d).exists();(base/d).symlink_to((old/d).resolve(),target_is_directory=True)
for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
 f=old/'data'/name;assert f.is_file();(base/'data'/name).symlink_to(f.resolve())
for name in ('official-model-manifest-original.json','predecessor-exit0.json','predecessor-teardown-closure.json'):
 with (base/'preparation'/name).open('xb') as f:f.write((old/'preparation'/name).read_bytes());f.flush();os.fsync(f.fileno())
config=json.loads((old/'preparation/h1008-launch-config.json').read_text())
config.update(base=str(base),source_commit=meta['commit'],official_manifest=str(base/'preparation/official-model-manifest-original.json'),frozen_teacher_cache=str(old/'teacher-warmup'),restore_receipt=None,
 new_loss=dict(loss_version='std_class_ce_v1',std_loss_weight=1.0,count_aux_loss_weight=.1,mean_loss_weight=0.0),superseded_training_base=str(old),superseded_job_id=3397)
cache=(old/'question-features').resolve();sha=hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest()
assert sha=='b59bc36032d54cd6359f644b9692875c31fbca73f129472e4d4dbe17e1c1235e'
(base/'question-features').symlink_to(cache,target_is_directory=True)
config['precomputed_question_features']=dict(path=str(cache),manifest_sha256=sha)
once(base/'preparation/h1008-launch-config.json',config);once(base/'preparation/source-files.json',meta['files'])
receipt=dict(base=str(base),source_commit=meta['commit'],verified_files=len(meta['files']),all_target_source_full_sha_match=True,submitted=False,model_optimizer_rollout_transferred_to_mac=False)
once(base/'preparation/deploy-receipt.json',receipt);print(json.dumps(receipt))
'''


def main():
    repo=Path(__file__).resolve().parents[2]
    proof=json.loads(Path(sys.argv[1]).read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
    assert proof['published_control_commit']==commit and proof['all_blobs_full_sha_match']
    archive=gzip.compress(subprocess.check_output(['git','archive',commit],cwd=repo))
    meta=dict(commit=commit,files=proof['files'],archive_bytes=len(archive),base='/mnt/data/shaoshitong/search-g1-qprobe3-stdloss-'+commit[:7]+'-20261015-v1')
    from reproduction.h1008_qprobe.management_transport import remote
    socket=json.loads(Path('work/qprobe3-management-current.json').read_text())['socket']
    header=json.dumps(meta).encode();payload=len(header).to_bytes(8,'big')+header+archive
    stage='/mnt/data/shaoshitong/qprobe3-code-staging-'+uuid.uuid4().hex
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix='outputs/gigpo-e1-instruct200/QPROBE3-CODE-'+stamp
    remote("from pathlib import Path;Path("+repr(stage)+").mkdir(mode=0o700)",prefix+'-claim.json',socket)
    chunks=[payload[i:i+32768] for i in range(0,len(payload),32768)]
    def send(item):
        index,data=item;digest=hashlib.sha256(data).hexdigest()
        script="import base64,hashlib;from pathlib import Path\ndata=base64.b64decode("+repr(base64.b64encode(data).decode())+")\nassert hashlib.sha256(data).hexdigest()=="+repr(digest)+"\nwith Path("+repr(stage+'/'+str(index))+ ").open('xb') as f:f.write(data)\nprint('source_chunk_sha_pass')"
        return remote(script,prefix+'-chunk'+str(index)+'.json',socket)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(send,enumerate(chunks)))
    assembly="import pathlib,hashlib\np=pathlib.Path("+repr(stage)+")\ndata=b''.join((p/str(i)).read_bytes() for i in range("+str(len(chunks))+"))\nassert hashlib.sha256(data).hexdigest()=="+repr(hashlib.sha256(payload).hexdigest())+"\n(p/'payload').write_bytes(data)\n"
    script=assembly+REMOTE.replace('stream=sys.stdin.buffer;',"stream=io.BytesIO(pathlib.Path("+repr(stage+'/payload')+").read_bytes());")
    print(remote(script,prefix+'-deploy.json',socket))


if __name__=='__main__':main()
