"""Scope native reservation, observed CPU and isolated QA costs without new remote work."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time


def duration_seconds(value):
    match=re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})',value)
    if match is None:
        raise ValueError('Unknown native runtime format: '+str(value))
    days,hours,minutes,seconds=map(int,(match[1]or'0',match[2],match[3],match[4]))
    if minutes>=60 or seconds>=60:
        raise ValueError('Invalid native runtime')
    return ((days*24+hours)*60+minutes)*60+seconds


def collect(root):
    root=Path(root);out=root/'outputs/search-g1-i1i6-fresh200-20261017'
    def read(name):return json.loads((out/name).read_text())
    inputs=['SOURCE3432_FINAL_ACTUAL.json','SOURCE3432_FINAL_RESOURCE.json','LATEST_ACTUAL.json',
        'QA5_STANDALONE_FAILURE1.json','QA5_STANDALONE_LATEST.json']
    source,current=read(inputs[0]),read(inputs[2]);run_id=current['latest_joint']['run_id']
    records=[];seen=set();peaks={};observed_cpu=0.
    for actual,resources in ((source,read(inputs[1])),(current,current['resource_summary'])):
        native=actual['native'];job=native['JobId']
        if (job in seen or job not in ('3432','3433') or native['Comment']!='g1-prefix200:'+run_id
                or native['WorkDir']!=actual['base'] or native['NodeList']!='h100-10'
                or native['NumCPUs']!='96' or native['TresPerNode']!='gres:gpu:8'
                or actual['latest_joint']['run_id']!=run_id):
            raise ValueError('Native reservation is not the actual same-run training chain')
        seen.add(job);seconds=duration_seconds(native['RunTime'])
        terminal=native['JobState'] in ('COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
        if (resources['job']!=job or resources['run_id']!=run_id or resources['base']!=actual['base']
                or resources['cpu_accounting_complete'] is not False):
            raise ValueError('Resource sample identity or incomplete-CPU scope changed')
        cpu=resources['latest']['owned_cpu_core_seconds_lower_bound'];observed_cpu+=cpu
        for uuid,row in resources['whole_card_peaks'].items():
            peak=peaks.setdefault(uuid,dict(max_used_mib=0,min_free_mib=None,max_utilization_percent=0))
            peak['max_used_mib']=max(peak['max_used_mib'],row['max_used_mib'])
            peak['min_free_mib']=row['min_free_mib']if peak['min_free_mib']is None else min(peak['min_free_mib'],row['min_free_mib'])
            peak['max_utilization_percent']=max(peak['max_utilization_percent'],row['max_utilization_percent'])
        records.append(dict(job_id=job,node=native['NodeList'],base=actual['base'],state=native['JobState'],
            native_observed_unix=actual['observed_unix'],native_runtime_seconds=seconds,terminal=terminal,
            reserved_gpu_count=8,reserved_cpu_count=96,gpu_reservation_hours=8*seconds/3600,
            cpu_reserved_capacity_hours=96*seconds/3600,
            observed_owned_cpu_core_seconds_lower_bound=cpu,resource_observed_unix=resources['last_observed_unix'],
            sampler_samples=resources['samples'],resource_coverage=resources['coverage'],
            resource_segments=resources.get('resource_segments'),
            resource_coverage_gaps=resources.get('resource_coverage_gaps',[]),
            cpu_lower_bound_combination=resources.get('cpu_lower_bound_combination'),
            start_time=native['StartTime'],end_time=native['EndTime'],
            role='planned_joint6_handoff'if job=='3432'else'current_same_run_continuation'))
    if seen!={'3432','3433'} or records[0]['end_time']>records[1]['start_time']:
        raise ValueError('Unexpected or overlapping native training chain')
    qa=[]
    for label,name in [('failed_shared_initialization','QA5_STANDALONE_FAILURE1.json'),
                       ('successful_shared_checkpoint5','QA5_STANDALONE_LATEST.json')]:
        evidence=read(name);terminal=evidence['terminal']
        if terminal['slurm_reserved_gpus']!=0 or terminal['outer']!=5:
            raise ValueError('Isolated QA cost changed resource scope')
        native_cpu=None
        unit=evidence.get('unit')
        if isinstance(unit,str):
            props=dict(line.split('=',1)for line in unit.splitlines()if'='in line)
            if props.get('CPUUsageNSec','').isdigit():native_cpu=int(props['CPUUsageNSec'])/1e9
        qa.append(dict(role=label,evidence=name,wall_seconds=terminal['ended_unix']-terminal['started_unix'],
            physical8_gpu_wall_opportunity_hours=terminal['wall_gpu8_hours'],slurm_gpu_reservation_hours=0,
            systemd_cpu_core_seconds=native_cpu,
            cpu_self_and_reaped_children_seconds=terminal['cpu_seconds_self_and_reaped_children'],
            success=terminal['success'],peak_sampled_own_gpu_memory_mib=terminal['peak_sampled_own_gpu_memory_mib']))
    value=dict(schema_version=1,run_id=run_id,created_unix=time.time(),actual_complete_outer=current['latest_joint']['outer'],
        native_training=records,total_native_gpu_reservation_hours=sum(r['gpu_reservation_hours']for r in records),
        total_native_cpu_reserved_capacity_hours=sum(r['cpu_reserved_capacity_hours']for r in records),
        observed_native_owned_cpu_core_seconds_lower_bound=observed_cpu,cpu_accounting_complete=False,
        whole_card_sampled_peaks=peaks,isolated_qa=qa,
        evidence_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest()for name in inputs},
        scope='Native reservation covers setup, useful updates, inline QA, retries and handoff; do not add phase wall again. Running-job runtime is a lower bound at its observation. CPU capacity is not CPU consumption. Observed owned CPU omits unobserved processes/time. Whole-card5s peaks include other users and are not continuous maxima. Isolated shared QA has zero Slurm reservation; physical8 wall opportunity is separate, not billed/exclusive GPU use. Historical invalid runs, remote acceptance/readers and missing pre-sampler CPU remain separate/unaccounted; no total end-to-end cost claim.')
    target=out/'RESOURCE_COST_CURRENT.json';target.write_text(json.dumps(value,indent=2)+'\n')
    for dest in (root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002'/out.name,
                 Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo')/out.name):
        dest.mkdir(parents=True,exist_ok=True);shutil.copy2(target,dest/target.name)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=collect(a.root)
    print(json.dumps({k:r[k]for k in ('actual_complete_outer','total_native_gpu_reservation_hours',
        'total_native_cpu_reserved_capacity_hours','observed_native_owned_cpu_core_seconds_lower_bound')}))
