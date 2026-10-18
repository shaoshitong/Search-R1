"""One foreground metadata check; no App scheduling or training mutation."""
import json
from pathlib import Path
import subprocess
import time

from reproduction.actor50_probe100.control import ROOT


def main():
    p=ROOT/'preparation/qprobe-teacher-warmup-a800-progress-handoff.json';h=json.loads(p.read_text())
    key='current_a800_actor50_prefix20_reuse20';assert h['current_authoritative']['active_experiment_key']==key
    c=h['current_authoritative'][key];workdir=Path(__file__).resolve().parents[2]
    r=subprocess.run(['python3','-m','reproduction.actor50_prefix20_reuse20.control','--action','observe','--handle',c['handle']],cwd=workdir,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=75)
    if r.returncode:raise RuntimeError('Observer unknown, not zero/exit: '+r.stderr[-2500:])
    wrapper=json.loads(r.stdout);out=ROOT/'preparation'/('actor50-prefix20-observation-wrapper-'+str(time.time_ns())+'.json');out.write_text(r.stdout)
    d=wrapper['result'];assert d['run_id']==c['run_id'] and d['source_commit']==c['source_commit']
    last=d['latest'];c.update(latest_actual=wrapper['receipt'],latest_actual_checked_unix=d['checked_unix'],pipeline_identity=d['leader'],native_status=d['status'],latest_failure=d['failure'],latest_cleanup=d['cleanup'],latest_exit=d['exit'])
    if last:
        assert last['actual_adam_steps_per_rank']==[last['ema_updates']]*4
        c.update(complete_epochs=last['completed_epochs'],actual_adam_updates=last['ema_updates'],ema_updates=last['ema_updates'],latest_complete_checkpoint=last)
    c['status']='COMPLETE%d_OF20'%(last['completed_epochs'] if last else 0)
    if d['complete']:c['status']='NATIVE_COMPLETE20_FINAL_AUDIT_PENDING'
    if d['failure']:c['status']='FAULT_KEEP_ACTUAL_RECEIPT_AND_COMPLETE_STATE'
    p.write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'preparation/actor50-prefix20-current-handoff.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(wrapper=str(out),receipt=wrapper['receipt'],checked_unix=d['checked_unix'],leader=d['leader'],stage=d['status'],
        epoch=last['completed_epochs'] if last else 0,Adam_EMA=last['ema_updates'] if last else 0,
        test_BA=last['metrics']['after']['test']['balanced_accuracy'] if last else None,
        train_BA=last['metrics']['after']['train']['balanced_accuracy'] if last else None,
        prepared_datasets=len(d['datasets']),features_completed_ranks=[x['feature_completed_ranks'] for x in d['datasets']],
        feature_current=[dict(dataset=x['dataset'],ranks=x['feature_progress']) for x in d['datasets'] if x['feature_completed_ranks']<4 and any(x['feature_progress'])],
        failure=d['failure'],exit=d['exit'],cleanup=d['cleanup'],complete=d['complete'] is not None)))


if __name__=='__main__':main()
