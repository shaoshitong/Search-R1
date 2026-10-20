"""Quiesce only this conflicting, zero-optimizer startup after human coordination."""
import json
from pathlib import Path
import subprocess
import time


def main():
    base=Path('/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp181-shared-recovery-20261020-r1')
    unit='sst-b-diverse-cp181-shared5-r1.service'
    read=lambda f:json.loads(f.read_text())
    receipt=read(base/'preparation/submit-receipt.json');who=receipt['leader']
    assert who['pid']==48221 and who['start_ticks']==613584
    assert receipt['source_commit']=='0f1bfd4fe2408f27fa86be566a8b824bc71834a4'
    proc=Path('/proc')/str(who['pid']);assert int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])==who['start_ticks']
    subprocess.run(['systemctl','freeze',unit],check=True,capture_output=True,text=True,timeout=20)
    try:
        # Check the writer while its entire own cgroup is frozen. Never stop
        # an optimizer batch using only a stale, pre-activation observation.
        attempts=list((base/'actor/outers').glob('outer-0182-attempt-*'))
        assert not attempts,'Optimizer stage already started; leave it untouched and reconcile'
        assert not(base/'outers/outer-000182/probe-update').exists()
        assert not(base/'outers/outer-000182/complete.json').exists()
        ready=read(base/'preparation/native-recovery-complete.json');assert ready['outer']==181 and ready['actor']==[3620]*8 and ready['probe']==[2530]*8
        services={}
        for name in(unit,'sst-b-diverse-cp181-shared5-r1-resources.service'):
            services[name]=subprocess.check_output(['systemctl','show',name,'-p','MainPID','-p','CPUUsageNSec','-p','ActiveState','-p','SubState'],text=True)
        value=dict(time=time.time(),status='COORDINATION_STOP_REQUESTED_ZERO_OPTIMIZER_STARTUP',human_message_id='01a156f4-210f-7db1-91ea-f52db916c7e8',reason='Avoid overlapping active peer B on5 GPUs4-7; instantaneous phase free memory is not sustainable capacity',native_leader=who,stage=read(base/'logs/pipeline-status.json'),actor_writes=0,probe_writes=0,retained_full_outer=181,services_before_stop=services,foreign_processes_signalled=False,user_goal_paused=False,exact_executor_resume_claimed=False)
        with(base/'preparation/coordination-zero-update-stop-intent.json').open('x')as h:json.dump(value,h)
        subprocess.run(['systemctl','stop','--no-block','sst-b-diverse-cp181-shared5-r1-resources.service',unit],check=True,capture_output=True,text=True,timeout=20)
    finally:
        subprocess.run(['systemctl','thaw',unit],capture_output=True,text=True,timeout=20)
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
