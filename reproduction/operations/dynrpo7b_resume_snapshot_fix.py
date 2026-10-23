"""Resume the serializer repair only after a verified full-boundary handoff.

This one-shot operation preserves the existing independent retriever. The
submission claim refuses any repeated or ambiguous submission.
"""
from reproduction.dynrpo7b.local_control import remote

SCRIPT = r'''
import subprocess, os
base='/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r6'
commit='ee74bfa3c6d1141859abb19e882b65c93ea6f0ba'
code=r"""
import json, pathlib, subprocess
from reproduction.dynrpo7b import deploy
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r6')
commit='ee74bfa3c6d1141859abb19e882b65c93ea6f0ba'
h=json.loads((b/'preparation/repair-handoff.json').read_text())
assert h['status']=='OLD_JOB_CANCEL_REQUESTED_FULL_STATE_RETAINED'
assert h['source_commit']==commit and h['completed_outer']>=5
state=subprocess.check_output(['sacct','-X','-j',str(h['old_job']),'-o','State','-P','-n'],text=True).strip()
assert state.startswith('CANCELLED'),state
assert not subprocess.check_output(['squeue','-h','-j',str(h['old_job']),'-o','%T'],text=True).strip()
proof=json.loads((b/'preparation/snapshot-serialization-cpu.json').read_text())
assert proof['status']=='REAL_PREFIX_SNAPSHOT_FIELDS_AND_RELOAD_IDENTICAL'
old=pathlib.Path(h['old_base'])
actual=[]
for rank in range(4):
 rows=[json.loads(s) for s in (old/'logs'/('optimizer-rank-%d.jsonl'%rank)).read_text().splitlines()]
 actual.append([s['write_count'] for s in rows if s.get('kind')=='OPTIMIZER_WRITE'][-1])
assert actual==h['actual_optimizer_writes_per_rank']
launch=deploy.launch_command
def exclude(*a,**k):
 c=launch(*a,**k);c.insert(1,'--exclude=ACD1-59');return c
deploy.launch_command=exclude
# Keep the tier already reached by the active run's measured queue escalation.
result=deploy.submit(b,commit,partition='emergency_acd',qos='normal',account='mou176',
 restore_receipt=h['restore_receipt'],external_retriever_receipt=str(b/'preparation/external-retriever.json'))
print(json.dumps(dict(submission=result,restored_outer=h['completed_outer'],actual_writes=actual)))
"""
p=subprocess.run([base+'/env/bin/python','-c',code],cwd=base+'/repo',
 env=dict(os.environ,QPROBE_SOURCE=base+'/source/'+commit),stdout=subprocess.PIPE,
 stderr=subprocess.PIPE,universal_newlines=True,timeout=120)
print(p.stdout);print(p.stderr[-2500:]);raise SystemExit(p.returncode)
'''

if __name__=='__main__':
    remote(SCRIPT)
