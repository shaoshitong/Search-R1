"""One bounded existing-metadata read; never launch, stop, evaluate or sample GPU."""
import json
from pathlib import Path
import subprocess
import time

from reproduction.actor50_probe100.control import ROOT


def main():
    path=ROOT/'preparation/qprobe-teacher-warmup-a800-progress-handoff.json';h=json.loads(path.read_text())
    key='current_a800_actor50_grpo_prefix2040';assert h['current_authoritative']['active_experiment_key']==key
    c=h['current_authoritative'][key];workdir=Path(__file__).resolve().parents[2]
    result=subprocess.run(['python3','-m','reproduction.actor50_grpo.control','--action','observe','--handle',c['handle']],
        cwd=workdir,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=75)
    if result.returncode:raise RuntimeError('Unknown observation; do not restart: '+result.stderr[-2000:])
    wrapper=json.loads(result.stdout);out=ROOT/'preparation'/('actor50-grpo-observation-wrapper-'+str(time.time_ns())+'.json');out.write_text(result.stdout)
    d=wrapper['result'];assert d['execution_id']==c['execution_id'] and d['source_commit']==c['source_commit']
    c.update(latest_actual=wrapper['receipt'],latest_actual_checked_unix=d['checked_unix'],
        stage=d['stage'],workers=d['workers'],latest_failure=d['failure'],latest_exit=d['exit'],latest_cleanup=d['cleanup'],actor_inflight=d.get('actor_inflight'))
    c.update(mixed_floor=d.get('mixed_floor'),mixed_floor_progress=d.get('mixed_floor_progress'),
        prepare_restart_exit=d.get('prepare_restart_exit'))
    j=d.get('joint')
    if j:
        assert len(set(j['actor_Adam']))==1 and j['scheduler']==j['outer']
        c.update(complete_outer=j['outer'],actor_actual_Adam_per_rank=j['actor_Adam'],
            new_actor_actual_updates_per_rank=[n-1000 for n in j['actor_Adam']],
            scheduler_completed_outer=j['scheduler'],probe=j['probe'],new_complete_outers=j['outer']-50)
        if j['outer']>50:c['CUDA_restore_verified']=True
    c['status']='GRPO_COMPLETE%d_OF200_RUNNING'%(j['outer'] if j else 50)
    if not d['workers']['pipeline'] and not d['complete']:c['status']='CPU_PREPARATION_OR_TERMINAL_INSPECT_ACTUAL_METADATA'
    if d['failure']:c['status']='ACTUAL_PIPELINE_FAILURE_REPAIR_REQUIRED'
    if d['complete']:c['status']='PIPELINE200_FINAL_METADATA_EXISTS_ACCEPTANCE_PENDING'
    path.write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'preparation/actor50-grpo-prefix20to40-current-handoff.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(receipt=wrapper['receipt'],at=d['checked_unix'],leader=d['workers']['pipeline'],
        stage=d['stage'],complete_outer=j['outer'] if j else None,actor_Adam=j['actor_Adam'] if j else None,
        probe=d['probe'],new_selected_Mixed=j['rollout_mixed'] if j and j['outer']>50 else None,
        actor_inflight=d.get('actor_inflight'),grounding_existing_logs=d.get('grounding_existing_logs'),advantage_summary=d.get('advantage_summary'),
        mixed_floor=d.get('mixed_floor'),mixed_floor_progress=d.get('mixed_floor_progress'),prepare_restart_exit=d.get('prepare_restart_exit'),
        rollout={phase:[None if x is None else x.get('completed_trajectories') for x in rows] for phase,rows in d.get('rollout',{}).items()},
        failure=d['failure'],exit=d['exit'],cleanup=d['cleanup'],complete=d['complete'])))


if __name__=='__main__':main()
