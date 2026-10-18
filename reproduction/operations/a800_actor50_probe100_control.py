"""Explicit human stop of the current formal run; retain immutable joint50.

Uses the established physical SSH route and deployed birth-authenticated pidfd
signal helper. No deletion, checkpoint rewriting, GPU probing or launch action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .a800_prefix5_variable_pool_control import route, ROOT


REMOTE = r'''
import hashlib,json,os,pathlib,signal,sys,time
sys.dont_write_bytecode=True
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'
assert h['run_id']=='54ac33dcebe64e62a11fe4177f41f116'
assert h['execution_id']=='fe7207faf3da498ea3b89bc38c3d6d63'
assert h['source_commit']=='eef31859ee20cf536e7c8e4d647306f92badba63'
assert str(b)=='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+h['run_id']+'-resume-'+h['execution_id']
total=0
def regular(path):
 path=pathlib.Path(path)
 assert path.resolve()==path and not path.is_symlink() and path.is_file(),str(path)
 return path
def read(path):
 global total
 path=regular(path);total+=path.stat().st_size;assert total<=64*1024**2
 return json.loads(path.read_text())
def sha(path):
 digest=hashlib.sha256()
 with regular(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
 return digest.hexdigest()
assert read(b/'preparation/deployment.json')==h
c=read(b/'preparation/launch-config.json')
for k in ('run_id','execution_id','source_commit','base'):assert c[k]==h[k]
for rel in ('reproduction/prefix_selection200/launch.py','reproduction/operations/a800_i1_i6_boundary_handoff.py'):
 assert sha(repo/rel)==h['source_files'][rel]
sys.path.insert(0,str(repo))
from reproduction.operations.a800_i1_i6_boundary_handoff import identity,same,send,owned_present
submit=read(b/'preparation/submit-receipt.json');who=identity(submit['leader'])
assert who==dict(pid=3143826,start_ticks=634612849,sid=3143826,uid=0)
assert who['uid']==os.getuid()
old=pathlib.Path('/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+h['run_id']+'-resume-a4a0c2a18cca40bc8c3850d40eed8554')
jointpath=old/'outers/outer-000050/complete.json'
assert sha(jointpath)=='80c5b4cdbafed1c92441352cd725cc7b2819319da37f7672498f58524ec79f2c'
j=read(jointpath);assert j['run_id']==h['run_id'] and j['outer']==50
assert j['actor']['completed_outer']==50 and j['actor']['actual_adam_steps_per_rank']==[1000]*4
assert j['probe']['completed_outer']==50 and j['probe']['actual_adam_steps_per_rank']==[250]*4
assert j['probe']['sha256']=='771f8c5c7b1a359cf786b5353a46261d23d2999567b3c8ba1bef3e0455c16184'
snapshot=regular(j['probe_snapshot']);assert snapshot.stat().st_size==j['probe']['bytes']
hf=pathlib.Path(j['actor']['hf_model_dir']);assert hf.resolve()==hf and hf.is_dir()
weights=[x for x in hf.iterdir() if x.is_file() and (x.name.endswith('.safetensors') or x.name.startswith('pytorch_model') and x.name.endswith('.bin'))]
assert weights and all(regular(x).stat().st_size>0 for x in weights)
proof=dict(joint_path=str(jointpath),joint_sha256=sha(jointpath),actor_metadata={k:v for k,v in j['actor'].items() if k not in ('reward','grounding_state')},
 probe_metadata=j['probe'],probe_snapshot=str(snapshot),probe_snapshot_bytes=snapshot.stat().st_size,
 actor_hf_weights=[dict(name=x.name,bytes=x.stat().st_size) for x in sorted(weights)],
 no_model_tensor_read=True,no_checkpoint_deleted_or_modified=True)
intent=b/'preparation/manual-stop-for-actor50-probe100-20261018.json'
before=same(who);signal_sent=False
if p['action']=='stop':
 assert p['human_intent']['scope']['stop_current_execution']==h['execution_id']
 assert p['human_intent']['scope']['preserve_actor50_and_probe50'] is True
 assert p['human_intent']['scope']['current_state_beyond50_save_required'] is False
 if not intent.exists():
  with intent.open('x') as stream:
   json.dump(dict(role='LATEST_HUMAN_STOP_CURRENT_FORMAL_KEEP_ACTOR50_PROBE50',human_intent_sha256=p['human_intent_sha256'],
    run_id=h['run_id'],execution_id=h['execution_id'],pipeline_identity=who,requested_unix=time.time(),
    preserve_joint50=proof['joint_sha256'],later_state_save_required=False),stream,sort_keys=True)
 if before and before['state'] not in ('Z','X'):
  send(who,signal.SIGTERM);signal_sent=True
 after=same(who)
else:after=before
present=owned_present(b)
raw=json.dumps(dict(status='HUMAN_STOP_SENT_NOT_TERMINAL' if signal_sent else 'EXISTING_STATE_OBSERVED',
 run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],checked_unix=time.time(),
 pipeline_identity=who,pipeline_before=before,pipeline_after=after,signal_sent=signal_sent,
 remaining_owned_birth_count=len(present),remaining_owned_active_birth_count=sum(x['state'] not in ('Z','X') for x in present),
 remaining_owned_births_sha256=hashlib.sha256(json.dumps(present,sort_keys=True).encode()).hexdigest(),
 cleanup=read(b/'logs/cleanup-status.json') if (b/'logs/cleanup-status.json').exists() else None,
 latest_saved_outer=read(b/'latest-joint.json')['outer'],protected_joint50=proof,
 metadata_bytes=total,no_model_tensor_read=True,no_deletion=True),allow_nan=False)
assert len(raw.encode())<=128*1024
print(raw)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('stop', 'inspect'), required=True)
    parser.add_argument('--handle', type=Path, required=True)
    parser.add_argument('--human-intent', type=Path, required=True)
    args = parser.parse_args()
    human_raw = args.human_intent.read_bytes()
    payload = dict(handle=json.loads(args.handle.read_text()), action=args.action,
                   human_intent=json.loads(human_raw),
                   human_intent_sha256=hashlib.sha256(human_raw).hexdigest())
    started = time.time()
    result = subprocess.run(route(REMOTE), input=json.dumps(payload).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=65)
    receipt = dict(action=args.action, checked_unix=time.time(), elapsed_seconds=time.time()-started,
                   operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   exit_code=result.returncode, stdout=result.stdout.decode(), stderr=result.stderr.decode())
    output = ROOT/'preparation'/('actor50-probe100-'+args.action+'-'+str(time.time_ns())+'.json')
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    if result.returncode or len(result.stdout)>128*1024:
        raise RuntimeError('Inspect receipt before retry: '+str(output))
    print(json.dumps(dict(receipt=str(output), result=json.loads(result.stdout))))


if __name__ == '__main__':
    main()
