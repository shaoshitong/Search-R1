"""Read actual new outer ledgers once; synchronize small local audit artifacts.

This entry has no submit/restart/stop, model loading, or training mutation path.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

from .poll_local import SCRIPT
from .watch_actual import EXTRA
from reproduction.h1008_qprobe.management_transport import remote

EXTRA_STATUS = r'''
value['config_sha256']=hashlib.sha256((base/'preparation/launch-config.json').read_bytes()).hexdigest()
value['qa_completed']=[]
for ordinal in (5,50,100,150,200):
 p=base/'outers'/('outer-%06d'%ordinal)/'validation/complete.json'
 if p.exists():
  qa=json.loads(p.read_text());assert qa['qa_count']==512
  value['qa_completed'].append(dict(outer=ordinal,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),result=qa))
p=base/'recovered-validation/complete.json'
if p.exists():
 qa=json.loads(p.read_text());ordinal=int(json.loads((base/'preparation/launch-config.json').read_text())['resume_joint']['completed_outer'])
 assert qa['actor_version'].endswith('-actor%04d'%ordinal) and qa['qa_count']==512
 value['qa_completed'].append(dict(outer=ordinal,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),result=qa))
p=base/'final-evaluation/complete.json'
value['final_qa']=json.loads(p.read_text()) if p.exists() else None
p=base/'preparation/resource5s-v1/summary.json'
value['resource_summary']=json.loads(p.read_text()) if p.exists() else None
'''

FIELDS = ['outer','actor_mixed_fraction','continued_mixed_fraction','actor_mean_em_variance',
 'continued_mean_em_variance','selection_mean_predicted_mixed','probe_train_accuracy',
 'probe_train_balanced_accuracy','probe_test_accuracy','probe_test_balanced_accuracy',
 'probe_loss_first','probe_loss_last','probe_updates_per_rank','actor_updates_per_rank','normal_phase_seconds']


def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def percentage(value):
    return 'undefined' if value is None else '%.2f%%' % (100*value)


def execution_base(joint):
    path=Path(joint['path'])
    return str(path.parent if path.name=='latest-joint.json' else path.parents[2])


def refresh(root):
    root=Path(root);work=root/'work/search-g1-prefix1024-diversity512-20261016'
    output=root/'outputs/search-g1-prefix1024-diversity512';config_path=work/'CURRENT_CONFIG.json'
    c=json.loads(config_path.read_text());series_path=output/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json'
    series=json.loads(series_path.read_text());verified=max(map(int,series))
    assert sorted(map(int,series))==list(range(1,verified+1))
    extra=EXTRA.replace('VERIFIED',str(verified))+EXTRA_STATUS
    code=SCRIPT.replace('print(json.dumps(value,allow_nan=False))',extra+'\nprint(json.dumps(value,allow_nan=False))').replace('BASE',repr(c['base']))
    transport=work/('refresh-'+str(time.time_ns())+'-transport.json')
    value=json.loads(remote(code,transport,target=c['node']))
    assert value['base']==c['base'] and value['config_sha256']==hashlib.sha256(config_path.read_bytes()).hexdigest()
    evidence=transport.with_name(transport.name.replace('-transport','-actual'));save(evidence,value)
    save(work/'LATEST_ACTUAL.json',value)
    native=value.get('native')or{};stage=value.get('stage')or{};joint=value['latest_joint'];n=joint['outer']
    if n<verified:raise ValueError('Actual boundary regressed; preserve prior records and explicitly reconcile recovery')
    for fact in value['new_complete_facts']:
        outer=fact['outer'];assert len(fact['actor'])==len(fact['probe'])==8
        for a in fact['actor']:
            assert a['writes']==20 and a['restore']['restored_scheduler']==outer-1 and a['last']['actual_adam_steps']==20*outer
        for p in fact['probe']:assert p['writes']==20 and p['last']['actual_adam_steps']==20*outer
        j=dict(run_id=c['run_id'],outer=outer,source_commit=c['source_commit'],observed_unix=fact['observed_unix'],
            path=c['base']+'/outers/outer-%06d/complete.json'%outer,sha256=fact['joint_sha256'],
            actor_adam=[a['last']['actual_adam_steps'] for a in fact['actor']],
            probe_adam=[p['last']['actual_adam_steps'] for p in fact['probe']],
            phase_timings=fact['phase_timings'],actor_selection=fact['actor_selection'],probe_metrics=fact['probe_metrics'])
        series[str(outer)]=dict(observed_unix=value['observed_unix'],joint=j,independent_direct_8rank_ledger_facts=True)
    assert sorted(map(int,series))==list(range(1,n+1))
    save(series_path,series)
    with (output/'ITERATION_METRICS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\n');writer.writeheader()
        for key,item in sorted(series.items(),key=lambda row:int(row[0])):
            j=item['joint'];a=j['actor_selection']['actor512'];b=j['actor_selection']['continued1024'];m=j['probe_metrics'];curve=m['global_step_loss_curve']
            vals=[int(key),a['mixed_fraction'],b['mixed_fraction'],a['mean_population_em_variance'],b['mean_population_em_variance'],
                a['mean_predicted_mixed_probability'],m['train']['accuracy'],m['train']['balanced_accuracy'],m['test']['accuracy'],
                m['test']['balanced_accuracy'],curve['first']['total_loss'],curve['last']['total_loss'],j['probe_adam'][0],j['actor_adam'][0],
                sum(p['elapsed_seconds']for p in j['phase_timings'].values())]
            writer.writerow(dict(zip(FIELDS,vals)))
    qa_path=output/'QA_MILESTONE_HISTORY.json';qa_history=json.loads(qa_path.read_text())if qa_path.exists()else{}
    if not qa_history and (output/'QA5_SUMMARY.json').exists():qa_history['5']=dict(outer=5,result=json.loads((output/'QA5_SUMMARY.json').read_text()))
    for qa in value['qa_completed']:
        result=qa['result'];assert result['run_id']==c['run_id'] and set(result['actual_worker_exit_codes'].values())=={0}
        qa_history[str(qa['outer'])]=qa
        save(output/('QA%d_SUMMARY.json'%qa['outer']),result)
    save(qa_path,qa_history);latest_qa=qa_history[max(qa_history,key=int)]if qa_history else None
    if value['final_qa']:save(output/'FINAL_QA_SUMMARY.json',value['final_qa'])
    resource=value['resource_summary']
    if resource:save(output/'RESOURCE5S_SUMMARY_LATEST.json',resource)
    intervals=[series[str(i)]['joint']['observed_unix']-series[str(i-1)]['joint']['observed_unix'] for i in range(max(7,n-3),n+1)
        if execution_base(series[str(i)]['joint'])==execution_base(series[str(i-1)]['joint'])
        and series[str(i)]['joint']['source_commit']==series[str(i-1)]['joint']['source_commit']]
    state_path=root/'outputs/STATUS.json';s=json.loads(state_path.read_text())
    actor=(value.get('actor')or{}).get('actual_adam_steps_per_rank');probe=(value.get('probe')or{}).get('actual_adam_steps_per_rank')
    selection=joint['actor_selection'];metrics=joint['probe_metrics']
    s.update(actual_complete_outer=n,stage=stage.get('stage'),native_state=native.get('JobState','UNKNOWN'),
        job_id=native.get('JobId',s['job_id']),active_training_job=native.get('JobId',s['job_id']),
        actual_actor_adam_per_rank=actor,actual_probe_adam_per_rank=probe,
        joint_actor_adam_per_rank=joint['actor_adam'],joint_probe_adam_per_rank=joint['probe_adam'],
        probe_saved_state_adam_per_rank=probe,actual_probe_physical_writes_per_rank=[p+h for p,h in zip(probe,c['historical_failed_unsaved_probe_writes_per_rank'])],
        latest_actual_observed_unix=value['observed_unix'],latest_actual_evidence=str(evidence),last_complete_joint_sha256=joint['sha256'],
        latest_actor512_mixed_fraction=selection['actor512']['mixed_fraction'],latest_continued1024_mixed_fraction=selection['continued1024']['mixed_fraction'],
        latest_probe_diagnostic_accuracy=metrics['test']['accuracy'],latest_probe_diagnostic_balanced_accuracy=metrics['test']['balanced_accuracy'],
        all_complete_outers_metric_rows=list(range(1,n+1)),recent_actual_whole_outer_seconds=intervals,
        recent_actual_whole_outer_mean_seconds=statistics.mean(intervals)if intervals else None,
        config_sha256=value['config_sha256'],updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_complete=False)
    if latest_qa:
        result=latest_qa['result'];s.update(latest_qa_outer=latest_qa['outer'],latest_qa_count=result['qa_count'],latest_qa_em=result['em'],latest_qa_f1=result['f1'])
    if resource:s.update(resource_sampler_samples_verified=resource['samples'],resource_sampler_cpu_accounting_complete=False)
    save(state_path,s);save(output/'STATUS.json',s)
    last=selection['actor512'];qa_text=('outer%d,512questions EM%.4f%% / F1%.4f%%'%(latest_qa['outer'],100*latest_qa['result']['em'],100*latest_qa['result']['f1']))if latest_qa else 'none verified'
    lines=['# Search-G1 predictedTop1024 / actualEM-varianceTop512','',
      'Actual complete%d/200; job%s %s on h100-10/10.33.0.18, stage %s. Saved actor/probe Adam per8rank: %s / %s. Completed joint counters: %s / %s. Historical failed unsaved probe20/rank remains separate; never sum8replicas as globalupdates.'%(n,s['job_id'],s['native_state'],s['stage'],actor,probe,joint['actor_adam'],joint['probe_adam']),
      '', 'Latest QA: '+qa_text+'. QA uses unchanged8TP1 FP32 greedy/full500/raw500/foursearch/noanswerreserve/noearlyfinal. Classification diagnostics below are distinct from QA EM/F1. Full final51713 remains required.','',
      'Latest actor512 mixed%.4f%%, meanEMvariance%.8f, selected frozen meanpredictedmixed%.8f. Continued1024 mixed%.4f%%; uncontinued4096 have no outcome labels.'%(100*last['mixed_fraction'],last['mean_population_em_variance'],last['mean_predicted_mixed_probability'],100*selection['continued1024']['mixed_fraction']),
      '', '|Complete outer|Continued1024 mixed|Actor512 mixed|Actor512 variance|Probe train ordinary / balanced|Probe fixed512 ordinary / balanced|',
      '|---|---:|---:|---:|---:|---:|']
    for key,item in sorted(series.items(),key=lambda row:int(row[0])):
        j=item['joint'];a=j['actor_selection'];m=j['probe_metrics'];lines.append('|%s|%s|%s|%.6f|%s / %s|%s / %s|'%(key,percentage(a['continued1024']['mixed_fraction']),percentage(a['actor512']['mixed_fraction']),a['actor512']['mean_population_em_variance'],percentage(m['train']['accuracy']),percentage(m['train']['balanced_accuracy']),percentage(m['test']['accuracy']),percentage(m['test']['balanced_accuracy'])))
    lines+=['','20-point probe losses and all6stagewall durations are in COMPLETED_ITERATION_SCALAR_EVIDENCE.json and ITERATION_METRICS.csv. Different natural batches mean first/last loss is not a monotonic same-batch fitting claim. Actor-selected allpositive balancedaccuracy is undefined when there is no negativeclass.',
      '', 'Protocol: officialQwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1; best actuallyretainedwarmupCP100 model-only probe initialization, no rewarmup.5120distinct diagnostic-excluded questions/G5prefix20, frozenactor/probe stable predictedmixedTop1024, continue authentic same5prefixes, population strictEMvarianceK(5-K)/25 selects512 intactG5 with frozenh/originalordinal ties. No extra sampling to force mixing. Actor originalSearch-G1 groundingsoftreward PPO20Adam/mini128/epoch1/globalmicro128 local16/noaccum, BF16compute/FP32masterAdam, LR1e-6/warm57outer. ProbeALL1024/5120 naturallyonce,20synchronousAdam/rank/globalbatch255/260avg256, LR1e-6, trajectorymeanBCE+ownG5mixedmeanBCE weights1/1. Fixeddiagnostic512 stays excluded. Only realtypedOOM permits preserving-budget micro fallback.',
      '', 'Frozen runtime'+c['source_commit']+'/actualconfigSHA'+value['config_sha256']+'. Later controls/results do not hotchange training. All models/native/optimizer/raw/fulloutputs stay sameH100cluster remote; local artifacts are small scalar/code summaries.',
      '', 'History:3416 preAdam pathfailure;3417 actor20saved/probe20unsaved diagnosticfailure (onlyprobe20recomputed/accounted);3418 savedjoint2 then imported-history prunerfailure;3421 savedjoint5 thenFP32H100TritonMMAcompilerassert inQA.3422 imported5 unchanged, compatiblefullprefillXFormers/CPATH fixedQA. Source5 independentfullSHA/CPU PASS:435FP32HFtensors,8nativeAdam/scheduler/RNG/driver/probe/pool; proofbe541a6076e1c2e0f2c022ba7eb2f31a10341bb0dc8d19e0e758904e61409159, acceptance5/. CPUload is not exactCUDAexecutorresume. Restored6–10+actual11generation accepted; failednative/log/birth/cost evidence retained. OldTree200/51713 is complete separately, not thisrun score.',
      '', 'Cost: COST_SNAPSHOT.json contains existing native token/request metadata and phasewall with explicit coverage; failed attempts/QA/setup/idle separate. Passive resource5s unit runs outside frozenrepo (CPU10%,Memory128M,Nice19), noGPUforward or trainingcontrol. Wholecard sampledpeaks cover only since22:49:59CST, not historical/continuouspeaks or amplecapacity. CPU observedbirth utime+stime is an incomplete lowerbound (freezeronly, past/shortlived CPU may be missing), never fullCPUcost. Raw telemetry staysremote. Fullretrieval and combinedGPU/CPU accounting remain outstanding.',
      '', 'Completion remains unproven: actual200, normal50/100/150/200checkpoint/QA, fullseven51713, HF/tokenizer/nativefinite/nonzeroupdate/source/split identity, cost/ownedcleanup and finalpublication. GoalACTIVE; no stop/restart is performed by this reader.']
    if intervals:lines.insert(5,'Recent actual same3422 completejoint intervals%s; mean%.3fs=%.3fmin. Startup,QA,failure/repair are separate.'%(intervals,statistics.mean(intervals),statistics.mean(intervals)/60))
    if resource:lines+=['','Latest passive telemetry%d samples, last observed%s; errors%s, wholecard maxused%dMiB/minfree%dMiB (coverage above).'%(resource['samples'],resource['last_observed_unix'],resource['latest']['errors'],max(x['max_used_mib'] for x in resource['whole_card_peaks'].values()),min(x['min_free_mib'] for x in resource['whole_card_peaks'].values()))]
    report='\n'.join(lines)+'\n';(output/'REPORT.md').write_text(report);(root/'outputs/REPORT.md').write_text(report)
    mirrors=[root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/search-g1-prefix1024-diversity512',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/search-g1-prefix1024-diversity512')]
    for destination in mirrors:
        destination.mkdir(parents=True,exist_ok=True)
        for name in ('STATUS.json','REPORT.md','ITERATION_METRICS.csv','COMPLETED_ITERATION_SCALAR_EVIDENCE.json','QA_MILESTONE_HISTORY.json','RESOURCE5S_SUMMARY_LATEST.json','FINAL_QA_SUMMARY.json')+tuple('QA%d_SUMMARY.json'%int(k)for k in qa_history):
            if (output/name).exists():shutil.copyfile(output/name,destination/name)
    return dict(evidence=str(evidence),native=s['native_state'],job=s['job_id'],complete_outer=n,stage=s['stage'],new_directly_verified=[f['outer']for f in value['new_complete_facts']],actor_saved=actor,probe_saved=probe,latest_qa=qa_text,resource_samples=resource['samples']if resource else None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();print(json.dumps(refresh(args.root),allow_nan=False))
