"""Read the existing post-200 pipeline; never launch, resume or stop a job.

Large joint metadata, rank evaluation rows and rollout shards stay on H100.
Only scalar summaries, bounded log tails and current process telemetry return.
"""
import datetime
import hashlib
import json
import time
from pathlib import Path

from .control import OUT, remote

PHYSICAL = '/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp188-four-h10011-20261020-r2'
CANONICAL = '/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/' + PHYSICAL.rsplit('/', 1)[1]
PID, START = 2580524, 725589138

SCRIPT = r'''
import pathlib,json,time,hashlib
p=pathlib.Path(BASE)
def read(f):
 try:return json.loads(f.read_text())
 except FileNotFoundError:return None
def identity(f):
 s=f.stat();return dict(dev=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def tail(f,limit=2000):
 with f.open('rb') as stream:
  stream.seek(max(0,f.stat().st_size-limit));return stream.read().decode(errors='replace')
def compact(value):
 if value is None:return None
 keys=('run_id','actor_version','actor_model_dir','qa_count','question_count','rank',
       'question_source_sha256','actual_compute_dtype','em','f1','per_source',
       'actual_worker_exit_codes','evaluation_contract','interaction_protocol',
       'completed_utc','full_output','output_shards','rank_receipts')
 return {k:value[k] for k in keys if k in value}
def evaluation(root):
 complete=read(root/'complete.json')
 ranks=[]
 for rank in range(4):
  rp=root/('rank-%d-complete.json'%rank);sealed=read(rp)
  progress=read(root/('rank-%d-progress.json'%rank))
  full=root/('rank-%d-full.jsonl.gz'%rank)
  log=root/('rank-%d.log'%rank)
  ranks.append(dict(rank=rank,complete=compact(sealed),progress=progress,
   completed_questions=sealed.get('qa_count') if sealed else (progress or{}).get('completed_questions',0),
   full_file_identity=identity(full) if full.exists() else None,
   log_tail=tail(log) if log.exists() else None))
 return dict(complete=compact(complete),ranks=ranks,
  completed_questions_lower_bound=sum(r['completed_questions'] for r in ranks))
c=read(p/'preparation/launch-config.json');receipt=read(p/'preparation/submit-receipt.json')
assert c['world_size']==4 and c['expected_execution_node']=='h100-11'
assert receipt['leader']['pid']==PID and receipt['leader']['start_ticks']==START
assert receipt['command'][-1]==CANONICAL
complete=p/'outers/outer-000200/complete.json'
assert identity(complete)==CACHE,'Sealed200 file changed; require a fresh full audit'
producer=p/'repo/reproduction/operations/b_replica_retained18/shared_resources.py'
assert hashlib.sha256(producer.read_bytes()).hexdigest()==RESOURCES_SHA
sample=None;resource=p/'logs/shared-resource-samples-5s.jsonl'
with resource.open('rb') as stream:
 stream.seek(max(0,resource.stat().st_size-32000));raw=stream.read()
for line in raw.splitlines(keepends=True):
 if line.endswith(b'\n'):
  try:sample=json.loads(line)
  except ValueError:pass
pipeline_path=p/'logs/pipeline-complete.json';pipeline=read(pipeline_path)
pipeline_summary=None
if pipeline is not None:
 # pipeline-complete embeds a very large latest_joint: never return it.
 pipeline_summary=dict(run_id=pipeline.get('run_id'),complete_outers=pipeline.get('complete_outers'),
  joint_outer=(pipeline.get('latest_joint')or{}).get('outer'),
  final_evaluation=compact(pipeline.get('final_evaluation')),identity=identity(pipeline_path))
logs=[]
for f in sorted((p/'logs').glob('*.log'),key=lambda f:f.stat().st_mtime,reverse=True)[:3]:
 logs.append(dict(path=str(f.relative_to(p)),identity=identity(f),tail=tail(f)))
fs=__import__('os').statvfs(p)
print(json.dumps(dict(observed_unix=time.time(),base=BASE,canonical=CANONICAL,
 source_commit=receipt['source_commit'],native_leader=receipt['leader'],complete_outer=200,
 joint_file_identity=CACHE,joint_file_identity_unchanged=True,
 stage=read(p/'logs/pipeline-status.json'),cleanup=read(p/'logs/cleanup-status.json'),
 qa_pending=read(p/'logs/qa-pending.json'),iteration_wall=read(p/'outers/outer-000200/iteration-wall-time.json'),
 validation200=evaluation(p/'outers/outer-000200/validation'),
 validation_retries={f.name:evaluation(f) for f in (p/'validation-retries').glob('*') if f.is_dir()},
 final51713=evaluation(p/'final-evaluation'),pipeline_complete=pipeline_summary,
 resource_latest=sample,resource_producer_sha256=RESOURCES_SHA,logs=logs,
 storage=dict(bavail_bytes=fs.f_bavail*fs.f_frsize,bfree_bytes=fs.f_bfree*fs.f_frsize,euid=__import__('os').geteuid()),
 training_modified=False)))
'''

HEALTH_SCRIPT = r'''
import pathlib,json,time,subprocess
p=pathlib.Path('/proc')/str(PID)
try:
 stat=(p/'stat').read_text();fields=stat.rsplit(')',1)[1].split()
 same=int(fields[19])==START and fields[0]!='Z'
except OSError:same=False
if same:assert BASE.encode() in (p/'cmdline').read_bytes().split(b'\0')
q=subprocess.run(['systemctl','show','sst-b-diverse-cp188-four-h10011-r2.service',
 '--property=ActiveState,SubState,MainPID,ExecMainStatus,Result,ControlGroup,CPUUsageNSec'],
 capture_output=True,text=True,timeout=15)
print(json.dumps(dict(observed_unix=time.time(),pid=PID,start_ticks=START,same_birth=same,
 systemd=q.stdout,systemd_returncode=q.returncode,source='direct_exact_birth_and_unit',training_modified=False)))
'''


def observe():
    shared = json.loads((OUT/'SHARED_RESUME64_LATEST.json').read_text())
    proof = json.loads((OUT/'CP200_COMPLETE_AUDIT_OBSERVATION1.json').read_text())
    assert shared['base'] == PHYSICAL and shared['complete_outer'] == 200
    assert proof['joint']['outer'] == 200 and proof['joint']['scheduler'] == 200
    fact = next(row for row in shared['facts'] if row['outer'] == 200)
    assert all(row['ledger_sha_matches'] for row in fact['ranks'])
    cache = fact['observation_cache']['complete_identity']
    producer_sha = hashlib.sha256(Path(__file__).with_name('shared_resources.py').read_bytes()).hexdigest()
    evidence = OUT/('post200-'+str(time.time_ns())+'-transport.json')
    prelude = '\n'.join(k+'='+repr(v) for k,v in dict(BASE=PHYSICAL,CANONICAL=CANONICAL,
        PID=PID,START=START,CACHE=cache,RESOURCES_SHA=producer_sha).items())
    value = json.loads(remote(prelude+'\n'+SCRIPT,evidence,target='h100-10',established_controller=True))
    sample = value.get('resource_latest')
    age = time.time()-(sample or{}).get('observed_unix',0)
    if sample and 0 <= age <= 20:
        value['actual_node_health'] = dict(observed_unix=sample['observed_unix'],pid=PID,
            start_ticks=START,same_birth=sample['leader_same_birth'],source='fresh_published_on_node_birth_sampler',
            telemetry_age_seconds=age)
    else:
        # The verified LAN alias is read-only and preserves our exact PID/base binding.
        from . import read_ip14_live
        original = read_ip14_live.SCRIPT
        try:
            read_ip14_live.SCRIPT = HEALTH_SCRIPT
            value['actual_node_health'] = read_ip14_live.read(CANONICAL,PID,START)
        finally:
            read_ip14_live.SCRIPT = original
    value['verified_training_joint_sha256'] = proof['joint']['sha256']
    path = evidence.with_name(evidence.name.replace('-transport','-actual'))
    value['evidence'] = str(path)
    path.write_text(json.dumps(value,indent=2))
    (OUT/'POST200_LATEST.json').write_text(json.dumps(value,indent=2))
    root = OUT.parents[1]
    status_path = root/'outputs/STATUS.json'
    status = json.loads(status_path.read_text())
    alive = value['actual_node_health']['same_birth']
    status.update(status='TRAINING200_POST_PHASES_CONTINUING' if alive else 'TRAINING200_NATIVE_TERMINAL_NEEDS_AUDIT',
        actual_complete_outer=200,actual_training_running=False,actual_native_running=alive,
        current_runtime_unknown=False,current_stage=(value.get('stage')or{}).get('stage'),current_evidence=str(path),
        post200_evidence=str(path),local_observer_session=None,training_observer_exit_code=0,
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2))
    for directory in (root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',
        Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018')):
        (directory/'POST200_LATEST.json').write_text(json.dumps(value,indent=2))
    return value


if __name__ == '__main__':
    value = observe()
    print(json.dumps(dict(observed_unix=value['observed_unix'],stage=value['stage'],
        health=value['actual_node_health'],validation200=value['validation200']['completed_questions_lower_bound'],
        validation200_complete=value['validation200']['complete'],
        final51713=value['final51713']['completed_questions_lower_bound'],
        final51713_complete=value['final51713']['complete'],
        pipeline_complete=bool(value['pipeline_complete']),cleanup=value['cleanup'],evidence=value['evidence'])))
