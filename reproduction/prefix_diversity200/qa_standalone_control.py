"""Small code/config-only isolated QA deployment; once-only systemd submission."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

OVERLAY=('__init__.py','actor.py','interaction.py','rollout.py','eval.py','evaluation_initialization.py','qa_standalone.py')


def deployment_script(config,commit):
    repo=Path(__file__).resolve().parents[2]
    files={name:subprocess.check_output(['git','-C',str(repo),'show',commit+':reproduction/prefix_diversity200/'+name]) for name in OVERLAY}
    payload=dict(config=dict(config,qa_source_commit=commit),source_commit=commit,files={name:base64.b64encode(value).decode() for name,value in files.items()},
        sha256={name:hashlib.sha256(value).hexdigest() for name,value in files.items()})
    return r'''
import base64,json,pathlib,hashlib,os,subprocess,time
p=PAYLOAD;c=p['config'];s=c['qa_standalone'];b=pathlib.Path(s['base'])
def read(x):return json.loads(pathlib.Path(x).read_text())
def sha(x):return hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest()
def once(x,v):
 with pathlib.Path(x).open('x') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
assert str(b).startswith('/mnt/shared/workspace/training_data_node_10/shaoshitong/search-g1-qa-i1i6-4a1cc2d5-outer0005-') and b.resolve()==b
assert c['run_id']=='4a1cc2d500fe40979ef8ca4792959438' and c['node']=='h100-5'
assert s['expected_count']==512 and s['outer']==5
if b.exists():
 print(json.dumps({'status':'EXISTING_QA_NAMESPACE_RECONCILE_NO_REPLAY','base':str(b)}));raise SystemExit
assert sha(s['source_evaluation_config'])==s['source_evaluation_config_sha256']
assert read(s['source_evaluation_config'])['actor_version']==c['actor_version']
assert sha(c['validation_questions'])==c['validation_questions_sha256']
source=pathlib.Path(s['source_cache']);corpus=read(source/'corpus.json');ready=read(source/'ready.json')
assert ready==s.pop('expected_source_retriever_ready') and ready['corpus']==corpus
assert sha(source/'corpus.json')==s['source_corpus_receipt_sha256']
assert read(pathlib.Path(s['arrow_directory'])/'dataset_info.json')['splits']['train']['num_examples']==s['corpus_documents']
assert len(s['arrow_shards'])==28
for ordinal,row in enumerate(s['arrow_shards']):
 shard=pathlib.Path(row['path']);assert shard.resolve()==shard and shard.parent==pathlib.Path(s['arrow_directory']) and shard.name=='json-train-%05d-of-00028.arrow'%ordinal and shard.stat().st_size==row['bytes']
s['source_retriever_ready_sha256']=sha(source/'ready.json')
archive=pathlib.Path(c['retrieval_corpus']);info=archive.stat()
assert info.st_size==corpus['source_size'] and info.st_mtime_ns==corpus['source_mtime_ns']
assert (source/'wiki_dump.jsonl').stat().st_size==corpus['content_bytes']
assert read(pathlib.Path(s['source_repo']).parent/'preparation/launch-config.json')['source_commit']=='33ef232e605f7864086d1685d9a4d8ea0d67f4ab'
b.mkdir(mode=0o700)
for name in ('preparation','logs','repo/reproduction/prefix_diversity200','retrieval-cache'):(b/name).mkdir(parents=True,exist_ok=True)
once(b/'preparation/deploy-intent.json',dict(source_commit=p['source_commit'],created_unix=time.time()))
for name,value in p['files'].items():
 data=base64.b64decode(value);assert hashlib.sha256(data).hexdigest()==p['sha256'][name]
 target=b/'repo/reproduction/prefix_diversity200'/name;target.write_bytes(data);assert sha(target)==p['sha256'][name]
# Same retained18 NFS inode; no model/data/index bytes pass through Mac.
os.link(source/'wiki_dump.jsonl',b/'retrieval-cache/wiki_dump.jsonl')
assert (source/'wiki_dump.jsonl').stat().st_ino==(b/'retrieval-cache/wiki_dump.jsonl').stat().st_ino
once(b/'retrieval-cache/corpus.json',dict(corpus,source=str(archive.resolve())))
once(b/'preparation/config.json',c);once(b/'preparation/source-files.json',p['sha256'])
result=dict(status='QA_CODE_CONFIG_DEPLOYED_NOT_STARTED',base=str(b),unit=s['unit'],config_sha256=sha(b/'preparation/config.json'),source_commit=p['source_commit'],overlay_sha256=p['sha256'],frozen_dependency_source=s['source_repo'],frozen_dependency_commit='33ef232e605f7864086d1685d9a4d8ea0d67f4ab',corpus_hardlink=True,large_assets_via_mac=False,training_modified=False,actual_config=c)
once(b/'preparation/deployment.json',result);print(json.dumps(result))
'''.replace('PAYLOAD',repr(payload))


def submit_script(base):
    return r'''
import json,pathlib,subprocess,os,time,fcntl
b=pathlib.Path(BASE);c=json.loads((b/'preparation/config.json').read_text());s=c['qa_standalone'];unit=s['unit']
assert str(b).startswith('/mnt/shared/workspace/training_data_node_10/shaoshitong/search-g1-qa-i1i6-4a1cc2d5-outer0005-') and b.resolve()==b and s['base']==str(b)
assert unit.startswith('g1-qa-4a1cc2d5-o5-') and unit.endswith('.service')
def once(p,v):
 with p.open('x') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
with (b/'preparation/submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (b/'preparation/submit-intent.json').exists():
  print(json.dumps({'status':'SUBMISSION_INTENT_EXISTS_RECONCILE_NO_REPLAY','unit':unit}));raise SystemExit
 status=subprocess.run(['systemctl','show',unit,'-p','LoadState','--value'],capture_output=True,text=True,timeout=10)
 assert status.stdout.strip()=='not-found',status.stdout
 command=['systemd-run','--unit='+unit,'--property=CPUQuota=3200%','--property=MemoryMax=256G','--property=KillMode=control-group','--property=TimeoutStopSec=60','--property=Restart=no','--property=StandardOutput=append:'+str(b/'logs/controller.log'),'--property=StandardError=inherit','--working-directory='+str(b/'repo'),'--setenv=PYTHONPATH='+str(b/'repo')+':'+s['source_repo']+':'+c['runtime_extra_packages'],str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.prefix_diversity200.qa_standalone','--config',str(b/'preparation/config.json')]
 once(b/'preparation/submit-intent.json',dict(unit=unit,command=command,created_unix=time.time(),slurm_reserved_gpus=0,physical_gpus=8))
 try:
  r=subprocess.run(command,capture_output=True,text=True,timeout=20);value=dict(unit=unit,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,status='SUBMITTED' if r.returncode==0 else 'REJECTED_NO_REPLAY')
 except subprocess.TimeoutExpired as e:value=dict(unit=unit,status='UNKNOWN_RECONCILE_NO_REPLAY',error=repr(e))
 once(b/'preparation/submit-receipt.json',value);print(json.dumps(value))
'''.replace('BASE',repr(base))
