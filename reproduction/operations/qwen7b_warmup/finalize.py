"""Seal full20-epoch delivery only after actual completed fit and full audits.

CPU postprocessing, run separately in an immutable auditors directory. Missing
training results are NOT_READY, never success. Never submits/cancels a job.
"""
import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import subprocess
import time
import audit_predictions as predictions
import audit_checkpoint as checkpoints

REVISION='a09a35458c702b33eeacc393d103063234e8bc28'


def read(path):return json.loads(path.read_text())


def require_ready(base):
    root=base/'probe'
    missing=[str(p.relative_to(base)) for p in [root/'complete.json',root/'recommended-ema.json']+
             [root/('epoch-%04d.json'%e) for e in range(1,21)] if not p.exists()]
    if missing:return dict(status='NOT_READY',missing=missing,goal_complete=False)
    job=read(base/'control/fit-submission.json')['job_id']
    queue=subprocess.check_output(['squeue','-h','-j',job,'-o','%i %T'],text=True)
    if queue.strip():return dict(status='NOT_READY',active_fit=queue.strip(),goal_complete=False)
    accounting=subprocess.check_output(['sacct','-n','-X','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],text=True)
    own=[line.split('|') for line in accounting.splitlines() if line.split('|')[0]==job]
    if not own:return dict(status='NOT_READY',accounting_unavailable=job,goal_complete=False)
    assert len(own)==1 and own[0][1]=='COMPLETED' and own[0][2]=='0:0',accounting
    return dict(status='READY',fit_job=job,sacct=accounting)


def verify_inputs(base):
    config=read(base/'config.json');bank=read(base/'bank/complete.json')
    assert config['model_revision']==REVISION and bank['model_revision']==REVISION
    for key,value in dict(feature_layer=26,native_head_layer=27,prefix_tokens=5,capture_prefix_tokens=20,
                          fit_epochs=20,fit_world=4,fit_lr=5e-6,fit_ema=.99,fit_group_weight=1.,
                          fit_global_groups=51,fit_micro_groups=2).items():assert config[key]==value,key
    assert config['warmup_only'] and not config['actor_updates_authorized'] and config['actor_optimizer_updates']==0
    assert config['common_manifest_sha256']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
    assert bank['status']=='FULL_CANONICAL_7B_G5_NATIVE_FEATURES_AND_OWN_EM_VERIFIED' and bank['actor_updates']==0
    assert (bank['train_questions'],bank['train_trajectories'],bank['diagnostic_questions'],bank['diagnostic_trajectories'])==(169028,845140,512,2560)
    for name in ('groups','features'):assert predictions.sha256(base/'bank'/(name+'.json'))==bank[name]['sha256']
    groups=read(base/'bank/groups.json');manifest=read(base/'inputs/manifest.json')
    assert len(groups['train'])==169028 and len(groups['diagnostic'])==512 and manifest['final_QA_used'] is False
    qids=[r['qid'] for rows in groups.values() for r in rows];assert len(set(qids))==169540
    import hashlib
    assert hashlib.sha256(json.dumps([r['qid'] for r in groups['train']]).encode()).hexdigest()==manifest['ordered_train_qid_sha256']
    for split,key in [('train','canonical_train'),('diagnostic','validation_questions')]:
        assert predictions.sha256(Path(config[key]))==config['input_sha256'][split]==manifest['source_sha256'][split]
    model=Path(config['model_dir']);migration=read(model/'complete.json')
    assert migration['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' and migration['fresh_all_files_SHA256'] and migration['revision']==REVISION
    for name,item in migration['files'].items():
        p=model/name;assert p.stat().st_size==item['bytes'] and predictions.sha256(p)==item['sha256']
    shapes=read(model/'config.json');assert shapes['hidden_size']==3584 and shapes['num_hidden_layers']==28
    source=read(base/'preparation/postprocess-source-publication.json')
    assert source['remote_sha']==source['commit']
    for name,sha in source['files'].items():assert predictions.sha256(Path(source['code_root'])/name)==sha
    proof=read(base/'probe-qualification/proof.json')
    assert proof['status']=='FOUR_RANK_GPU_RECOVERY_QUALIFICATION_PASSED' and proof['production_optimizer_updates']==0
    assert proof['real_prefix5_future15_invariance'] and proof['real_uneven_rank_evaluation_groups']==51
    assert proof['fit_code_sha256']==predictions.sha256(Path(source['code_root'])/'reproduction/operations/qwen7b_warmup/train.py')
    assert [r['rank'] for r in proof['rank_results']]==list(range(4))
    assert all(r['production_unchanged'] and all(r[k]['passed'] for k in ('raw','ema','Adam')) for r in proof['rank_results'])
    shards=[read(base/'sampling'/('shard-%02d'%i)/'complete.json') for i in range(8)]
    assert all(s['run_id']==bank['run_id'] and s['model_revision']==REVISION for s in shards)
    assert sum(s['questions'] for s in shards)==169540 and sum(s['trajectories'] for s in shards)==847700
    assert config['sampling_shards']==8 and config['gpus_per_sampling_job']==2
    return config,bank,groups,source,proof


def check_curve(base,plans):
    expected={e:sum(len(plan) for plan in plans[:e]) for e in range(1,21)}
    count=0
    with (base/'probe/curve.jsonl').open() as handle:
        for line in handle:
            row=json.loads(line);count+=1;epoch=row['epoch']
            assert 1<=epoch<=20 and row['steps']==row['ema_steps']==count and row['actor_updates']==0
            before=expected[epoch-1] if epoch>1 else 0
            assert row['batch']==count-before and row['batches']==len(plans[epoch-1])
            assert all(math.isfinite(row[k]) and row[k]>=0 for k in ('trajectory_bce','group_bce','total_loss'))
    assert count==expected[20]
    return count,expected


def plot_curves(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for axis,key,title,scale in [(axes[0,0],'total_loss','Joint loss',1),
                                (axes[0,1],'group_bce','Mixed BCE',1),
                                (axes[1,0],'trajectory_accuracy','Trajectory accuracy (%)',100),
                                (axes[1,1],'mixed_balanced_accuracy','Mixed balanced accuracy (%)',100)]:
        for split in ('train','diagnostic'):
            axis.plot([r['epoch'] for r in rows],[r[split][key]*scale for r in rows],marker='.',label=split)
        axis.set(xlabel='Completed epoch',ylabel=title);axis.grid(alpha=.25);axis.legend()
    fig.suptitle('Official Qwen2.5-7B frozen actor; layer26 -> block27; Prefix5; EMA0.99')
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True,type=Path);parser.add_argument('--check-ready',action='store_true');args=parser.parse_args()
    base=args.base;ready=require_ready(base)
    if ready['status']!='READY' or args.check_ready:
        print('RESULT_JSON='+json.dumps(ready),flush=True);return
    output=base/'delivery';output.mkdir(exist_ok=True)
    lock=(output/'finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # No actor or teacher is instantiated. Checkpoints are CPU mmap reads.
    config,bank,groups,source,recovery=verify_inputs(base)
    from reproduction.operations.qwen7b_warmup.train import balanced_plan
    plans=[];coverage=[]
    for epoch in range(1,21):
        plan,info=balanced_plan(groups['train'],config['seed'],epoch);flat=[i for batch in plan for i in batch]
        assert len(set(flat))==169028 and len(flat)==info['groups']
        plans.append(plan);coverage.append(dict(epoch=epoch,unique_questions=len(set(flat)),sampled_questions=len(flat),
            repeated_questions=len(flat)-len(set(flat)),optimizer_updates=len(plan),**info))
    total_steps,expected=check_curve(base,plans);rows=[];checkpoint_reports=[];prediction_reports=[]
    for epoch in range(21):
        receipt=read(base/'probe'/('epoch-%04d.json'%epoch)) if epoch else None
        metrics=receipt['metrics'] if receipt else read(base/'probe/epoch-0000-metrics.json')
        accepted={split:predictions.accept_split(base,epoch,split,groups[split],metrics[split]) for split in ('train','diagnostic')}
        prediction_reports.append(dict(epoch=epoch,splits=accepted))
        rows.append(dict(epoch=epoch,steps=expected[epoch] if epoch else 0,train=metrics['train'],diagnostic=metrics['diagnostic']))
        if receipt:
            assert receipt['completed_epoch'] and receipt['cursor']==0 and receipt['steps']==expected[epoch]
            assert metrics['balance']['groups']==coverage[epoch-1]['sampled_questions']
            checkpoint_reports.append(checkpoints.accept(base,receipt))
        print('ACCEPTED_EPOCH='+str(epoch),flush=True)
    complete=read(base/'probe/complete.json');assert complete['status']=='FULL_DATA_7B_PROBE_WARMUP20_COMPLETE'
    assert complete['epochs']==20 and complete['steps']==total_steps and complete['actor_updates']==0
    eligible=rows[1:];best=max(eligible,key=lambda r:(r['diagnostic']['mixed_balanced_accuracy'],-r['diagnostic']['total_loss'],-r['epoch']))
    recommended=read(base/'probe/recommended-ema.json');assert recommended['epoch']==best['epoch'] and recommended['weight_key']=='ema'
    assert recommended['sha256']==read(base/'probe'/('epoch-%04d.json'%best['epoch']))['sha256']
    selection='maximum independent diagnostic Mixed BA, then minimum joint loss, then earlier epoch; epochs1..20 only'
    report=dict(status='FULL20_NATIVE7B_WARMUP_DELIVERY_VERIFIED',model_revision=REVISION,fit_source_commit=source['commit'],
        total_epochs=20,total_optimizer_updates=total_steps,actor_updates=0,bank=bank,fit_job=ready,
        coverage=coverage,epochs=rows,recommended_epoch=best['epoch'],recommended_checkpoint=recommended,
        selection=selection,checkpoint_acceptance=checkpoint_reports,prediction_acceptance=prediction_reports,
        recovery_qualification=recovery,epoch_time=time.time())
    predictions.atomic(output/'metrics-and-acceptance.json',report)
    columns=['epoch','steps']+[split+'_'+key for split in ('train','diagnostic') for key in ('trajectory_bce','group_bce','total_loss','trajectory_accuracy','mixed_balanced_accuracy')]
    with (output/'epoch-curves.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow(dict(epoch=row['epoch'],steps=row['steps'],**{split+'_'+key:row[split][key] for split in ('train','diagnostic') for key in ('trajectory_bce','group_bce','total_loss','trajectory_accuracy','mixed_balanced_accuracy')}))
    plot_curves(rows,output/'epoch-curves.png')
    lines=['# Qwen2.5-7B native probe warmup: verified20-epoch delivery','',
        'Frozen official actor; zero-based layer26 post-block native features -> layer27 block + Linear. Prefix5 fit; same-trajectory Prefix20 retained. No student RL.',
        '',f'Fit source: {source["commit"]}; total genuine Adam updates: {total_steps}.',
        f'Recommended EMA: epoch{best["epoch"]}; checkpoint: {recommended["path"]}; key=ema.',
        'Selection: '+selection+'.','',
        '| Epoch | Adam steps | Train joint loss | Diagnostic joint loss | Diagnostic trajectory accuracy | Diagnostic Mixed BA |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append('| %d | %d | %.6f | %.6f | %.2f%% | %.2f%% |'%(row['epoch'],row['steps'],row['train']['total_loss'],row['diagnostic']['total_loss'],100*row['diagnostic']['trajectory_accuracy'],100*row['diagnostic']['mixed_balanced_accuracy']))
    lines+=['','Epoch0 is initialization, not a trained checkpoint. Training-label distribution and class-balanced batch losses are distinct from natural diagnostic performance.',
        'Three-class confusion in the original metrics thresholds each p_j at0.5 and counts predicted correct trajectories; binary Mixed metrics threshold the group-event probability at0.5.',
        'All full G5 prediction identities, own EM, probabilities and confusion metrics were independently streamed; all20 full checkpoints were SHA/CPU/rank Adam/RNG accepted. Actual GPU next-update recovery proof is included.',
        'All assets, features, predictions and checkpoints remain remote. The final51713 QA set was excluded. Slurm fit is COMPLETED0:0 and no training process was stopped by this finalizer.','',
        '![Epoch curves](epoch-curves.png)']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    artifacts={name:dict(bytes=(output/name).stat().st_size,sha256=predictions.sha256(output/name)) for name in ('metrics-and-acceptance.json','epoch-curves.csv','epoch-curves.png','REPORT.md')}
    predictions.atomic(output/'complete.json',dict(status='FULL20_TRAINING_PREDICTIONS_CHECKPOINTS_AND_REPORT_VERIFIED',epochs=20,steps=total_steps,
        recommended_epoch=best['epoch'],artifacts=artifacts,actor_updates=0,fit_job=ready,epoch_time=time.time()))
    print('RESULT_JSON='+json.dumps(dict(status='DELIVERY_COMPLETE',path=str(output),recommended_epoch=best['epoch'])),flush=True)


if __name__=='__main__':main()
