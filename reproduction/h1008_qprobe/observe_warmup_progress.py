"""Read-only scalar progress; run on the training host, never return trajectories."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def rows(path):
    result=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:result.append(json.loads(line))
            except ValueError:pass  # a currently appending final line
    return result


def snapshot(base, job):
    run=read(base/'logs/run-manifest.json')
    probe=base/'artifacts/question-probe'
    curve=rows(probe/'teacher-warmup-training-curve.jsonl')
    epoch=[row for row in curve if row.get('event')=='epoch']
    updates=rows(base/'logs/actual-updates.jsonl')
    keys=('epoch','optimizer_update','replay_weighted_loss','binary_accuracy',
          'binary_precision','binary_recall','binary_f1','binary_bce','binary_brier',
          'binary_majority_accuracy_baseline','p_mix_mean','p_mix_min','p_mix_max',
          'p_mix_top_mixed_rate','random_mixed_rate_expected','variance_mse','variance_normalized_mse')
    ledger=read(base/'logs/process-ledger.json');owned=[]
    for name,value in ledger.get('jobs',{}).items():
        for birth in value.get('ledger',{}).values():
            stat=Path('/proc')/str(birth['pid'])/'stat'
            if stat.exists():
                fields=stat.read_text().rsplit(')',1)[1].split()
                if fields[19]==str(birth['start_ticks']):
                    owned.append(dict(component=name,pid=birth['pid'],
                        cpu_ticks=int(fields[11])+int(fields[12])))
    errors=[]
    for path in (base/'logs').glob('*.log'):
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-16384));tail=stream.read().decode(errors='replace')
        errors += [line[-1500:] for line in tail.splitlines() if line.startswith(
            ('AssertionError','ValueError','RuntimeError','TypeError','KeyError','Traceback'))]
    native=subprocess.run(['scontrol','show','job','-o',str(job)],capture_output=True,text=True)
    fields={x.split('=',1)[0]:x.split('=',1)[1] for x in native.stdout.split() if '=' in x}
    rank_writes=[]
    for rank in range(8):
        records=[r for r in rows(base/f'logs/optimizer-rank-{rank}.jsonl') if r.get('kind')=='OPTIMIZER_WRITE']
        rank_writes.append(records[-1]['write_count'] if records else 0)
    ready=read(probe/'teacher-warmup-ready.json')
    init=read(probe/'gpu-worker-initialization.json')
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=job,
        native={k:fields.get(k) for k in ('JobState','NodeList','ExitCode','Reason')},
        run_id=run.get('run_id'),source_commit=run.get('source_commit'),
        pipeline=read(base/'logs/pipeline-status.json').get('stage'),
        pipeline_error=read(base/'logs/pipeline-status.json').get('error'),
        warmup_config_epochs=run.get('question_probe',{}).get('teacher_warmup_epochs'),
        before0_repeated=run.get('pretraining_baseline_policy')!='reuse_historical' if run else None,
        warmup_adam_updates=curve[-1].get('optimizer_update',0) if curve else 0,
        warmup_completed_epochs=max((r['epoch'] for r in epoch),default=0),
        epoch_metrics=[{k:r[k] for k in keys if k in r} for r in epoch],
        warmup_ready_status=ready.get('status'),
        warmup_fit_actual_updates=ready.get('head_metrics',{}).get('head_optimizer_updates'),
        actual_probe_rank=init.get('worker_rank'),actual_probe_gpu_uuid=init.get('gpu_uuid'),
        complete_actor_outer=max((r.get('step',0) for r in updates),default=0),
        actual_actor_adam_per_rank=rank_writes,owned_live_processes=len(owned),
        trainer_cpu_ticks=sum(r['cpu_ticks'] for r in owned if r['component']=='trainer'),
        owned_component_cpu=sorted((r for r in owned if r['component']=='trainer'),
            key=lambda r:r['cpu_ticks'],reverse=True)[:8],errors=errors[-6:])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--job',type=int,required=True);parser.add_argument('--samples',type=int,default=2)
    parser.add_argument('--interval',type=float,default=20);args=parser.parse_args()
    assert str(args.base).startswith(('/mnt/data/shaoshitong/search-g1-qprobe3-warmup10-',
        '/mnt/data/shaoshitong/search-g1-qprobe3-binary10-'))
    assert 1<=args.samples<=3 and 0<=args.interval<=30
    for index in range(args.samples):
        if index:time.sleep(args.interval)
        print(json.dumps(snapshot(args.base,args.job)),flush=True)


if __name__=='__main__':main()
