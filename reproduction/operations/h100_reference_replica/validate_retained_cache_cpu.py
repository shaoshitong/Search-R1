"""Actual frozen corpus/Arrow cache entry in a fresh retained namespace; CPU only."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

INNER=r'''
import hashlib,json,sys,time
from pathlib import Path
BINDING=__BINDING__
REPO=__REPO__
b=Path(BINDING);proof=json.loads((b/'RESULT.json').read_text());cache=Path(proof['virtual_cache'])
if hashlib.sha256((cache/'corpus.json').read_bytes()).hexdigest()!=proof['private_receipt_sha256']:raise ValueError('Private receipt changed')
sys.path.insert(0,REPO)
from reproduction.prefix20_online.retriever import prepare_corpus
started=time.monotonic();corpus,extraction=prepare_corpus(proof['virtual_archive'],str(cache))
if extraction!=proof['new_receipt']:raise ValueError('Frozen prepare_corpus did not load exact relocated receipt')
import datasets
builder=datasets.load_dataset_builder('json',data_files=str(corpus),cache_dir=str(cache/'arrow'))
if not (Path(builder.cache_dir)/'dataset_info.json').is_file():raise ValueError('Existing Arrow fingerprint missing; do not regenerate 14GB data')
data=datasets.load_dataset('json',data_files=str(corpus),split='train',num_proc=4,cache_dir=str(cache/'arrow'))
expected=json.loads((cache/'ready.json').read_text())['documents']
if len(data)!=expected:raise ValueError('Cached document count differs from original ready receipt')
files={r['relative']:r for r in proof['files']};validated=[]
for row in data.cache_files:
 p=Path(row['filename']);relative=p.relative_to(cache).as_posix();old=files[relative];s=p.stat()
 if [s.st_ino,s.st_size,s.st_mtime_ns]!=old['identity'][1:]:raise ValueError('Cached Arrow payload identity changed')
 validated.append(dict(relative=relative,bytes=s.st_size,prior_full_sha=old['prior_full_sha']))
print(json.dumps(dict(status='FROZEN_CORPUS_PREPARE_AND_EXISTING_ARROW_CPU_ENTRY_PASS',checked_unix=time.time(),documents=len(data),arrow_shards=len(validated),arrow_files=validated,cache_dir=builder.cache_dir,private_receipt_sha256=proof['private_receipt_sha256'],frozen_retriever_source_sha256=hashlib.sha256((Path(REPO)/'reproduction/prefix20_online/retriever.py').read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,gpu_forward_performed=False,large_payload_hash_repeated=False,json_regeneration_performed=False,live_training_changed=False)))
'''

def run(binding,repo):
    b=Path(binding);output=b/'CPU_ENTRY_RESULT.json'
    if output.exists():return json.loads(output.read_text())
    wrapper=b/'retained_cache_overlay_paths.sh';source=INNER.replace('__BINDING__',repr(str(b))).replace('__REPO__',repr(str(repo)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_DATASETS_OFFLINE='1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    p=subprocess.run(['unshare','--mount','--propagation','private','bash',str(wrapper),'/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',source],env=env,text=True,capture_output=True,timeout=80)
    if p.returncode:raise RuntimeError(p.stderr[-4000:])
    value=json.loads(p.stdout)
    with output.open('x')as f:json.dump(value,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
    return value
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--repo',required=True);a=p.parse_args();print(json.dumps(run(a.binding,a.repo)))
