"""One scalar observation of the actual fresh Prefix20 continuation."""
import json
from pathlib import Path
import subprocess
import time

from reproduction.actor50_probe100.control import ROOT


def main():
    hp=ROOT/'preparation/qprobe-teacher-warmup-a800-progress-handoff.json';h=json.loads(hp.read_text())
    key='current_a800_actor50_prefix20_fresh100';assert h['current_authoritative']['active_experiment_key']==key
    c=h['current_authoritative'][key];workdir=Path(__file__).resolve().parents[2]
    r=subprocess.run(['python3','-m','reproduction.actor50_prefix20_fresh100.control','--action','observe','--handle',c['handle']],
        cwd=workdir,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=75)
    if r.returncode:raise RuntimeError('Observer unknown, not zero or exited: '+r.stderr[-2000:])
    wrapper=json.loads(r.stdout);out=ROOT/'preparation'/('actor50-prefix20-fresh-observation-wrapper-'+str(time.time_ns())+'.json');out.write_text(r.stdout)
    d=wrapper['result'];last=d['latest'];assert d['run_id']==c['run_id'] and d['source_commit']==c['source_commit']
    c.update(latest_actual=wrapper['receipt'],latest_actual_checked_unix=d['checked_unix'],pipeline_identity=d['leader'],native_status=d['status'],
        latest_failure=d['failure'],latest_cleanup=d['cleanup'],latest_exit=d['exit'],fresh_plan=d['fresh_plan'],pool_cursor=d['pool_cursor'])
    if last:
        assert last['actual_adam_steps_per_rank']==[last['ema_updates']]*4
        c.update(complete_epochs=last['completed_epochs'],actual_adam_updates=last['ema_updates'],ema_updates=last['ema_updates'],latest_complete_checkpoint=last)
        if last['completed_epochs']>c['resume_completed_epochs']:
            c['actual_CUDA_continuation_verified']=True;c['actual_CUDA_continuation_receipt']=wrapper['receipt']
    c['status']='COMPLETE%d_OF100_FRESH_NATIVE20'%(last['completed_epochs'] if last else c['resume_completed_epochs'])
    if d['failure']:c['status']='FAULT_KEEP_ACTUAL_COMPLETE_STATE'
    if d['complete']:c['status']='COMPLETE100_NATIVE_FINAL_AUDIT_PENDING'
    hp.write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'preparation/actor50-prefix20-fresh100-current-handoff.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(wrapper=str(out),receipt=wrapper['receipt'],at=d['checked_unix'],leader=d['leader'],stage=d['status'],
        epoch=last['completed_epochs'],Adam_EMA=last['ema_updates'],test_BA=last['metrics']['after']['test']['balanced_accuracy'],
        test_trajectory_accuracy=last['metrics']['after']['test']['trajectory_accuracy'],train_BA=last['metrics']['after']['train']['balanced_accuracy'],
        fresh_plan=d['fresh_plan'],pool_cursor=d['pool_cursor'],
        rollout={k:[None if x is None else x['completed_trajectories'] for x in v] for k,v in (d.get('rollout_progress') or {}).items()},
        actual_steps_inflight=[None if x is None else x['actual_returned_adam'] for x in (d.get('rank_progress') or [])],
        failure=d['failure'],exit=d['exit'],cleanup=d['cleanup'],complete=d['complete'] is not None)))


if __name__=='__main__':main()
