"""Resume the retained outer4 collection after the terminal zero-write R4 failure."""
from reproduction.dynrpo7b.local_control import remote

SCRIPT = r'''
import subprocess,pathlib,os
base='/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r5'
commit='3f1650d477daa5c4f3bbae568175bf8638b9d831'
code=r"""
import pathlib,json,hashlib,subprocess
from reproduction.dynrpo7b.deploy import prepare,publish,submit
from reproduction.dynrpo7b import deploy
from reproduction.dynrpo7b.collection_resume import find_resume
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r5')
old=pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r4')
original=pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023')
commit='3f1650d477daa5c4f3bbae568175bf8638b9d831'
assert json.loads((base/'preparation/retained-outer4-probability-cpu.json').read_text())['accepted']
state=subprocess.check_output(['sacct','-X','-j','751532','-o','State','-P','-n'],text=True).strip()
assert state=='FAILED',state
receipt=original/'preparation/dynrpo7b-checkpoints/global_step_3/boundary.json'
cp=json.loads(receipt.read_text())
assert cp['completed_outer']==3
actual=json.loads((old/'logs/full-state-restored-workers.json').read_text())['ranks']
assert len(actual)==4 and all(r['actual_optimizer_writes']==60 and r['actual_adam_scheduler_readback_equal'] for r in actual)
pointer=old/'preparation/dynrpo7b-collection/outer-0004.json'
run=json.loads((old/'logs/run-manifest.json').read_text())
collection=find_resume(old,run,4,60)
assert collection['stage']=='collected'
prepare(base,commit,predecessor=str(original))
external=json.loads((old/'preparation/external-retriever.json').read_text())
publish(base/'preparation/external-retriever.json',external)
pin=dict(source_base=str(old),outer=4,pointer_sha256=hashlib.sha256(pointer.read_bytes()).hexdigest(),source_job=751532,terminal_state=state,actor_restore_receipt=str(receipt),prior_writes_per_rank=[60]*4)
publish(base/'preparation/collection-resume-origin.json',pin)
assert find_resume(base,run,4,60)==collection
launch=deploy.launch_command
def exclude(*a,**k):
 c=launch(*a,**k);c.insert(1,'--exclude=ACD1-59');return c
deploy.launch_command=exclude
result=submit(base,commit,partition='acd_ue',qos='normal',account='mou176',restore_receipt=str(receipt),external_retriever_receipt=str(base/'preparation/external-retriever.json'))
print(json.dumps(dict(submission=result,retained_collection=pin)))
"""
env=dict(os.environ,QPROBE_SOURCE=base+'/source/'+commit)
r=subprocess.run(['/data/user/jxu957/experiments/dynrpo7b-formal-20261023/env/bin/python','-c',code],cwd=base+'/source/'+commit,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=140)
print(r.stdout);print(r.stderr[-2500:]);raise SystemExit(r.returncode)
'''

if __name__ == "__main__":
    remote(SCRIPT)

