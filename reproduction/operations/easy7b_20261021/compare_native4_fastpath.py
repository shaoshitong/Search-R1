"""Compare the small reports from independent legacy/leancompact torchrun4 jobs.

Stdlib-only, read-only: no CUDA, SSH, Slurm, tensors, model or optimizer loading.
The numerical comparison covers reported LP/parameter/Adam samples, not all
parameter or optimizer elements, and never establishes whole-outer speedup.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):return json.loads(Path(path).read_text())


def reports(path):
    path=Path(path).resolve();root=path if path.is_dir() else path.parent
    result_path=root/'result.json' if path.is_dir() else path
    result=read(result_path) if result_path.exists() else None
    values={}
    for member in sorted(root.glob('rank-*.json')):
        row=read(member)
        if 'rank' in row and 'timings' in row:values[int(row['rank'])]=row
    if isinstance(result,dict) and 'rank' in result and 'timings' in result:
        values[int(result['rank'])]=result
    return dict(root=str(root),result=result,ranks=values)


def numbers(values):
    result=[]
    for value in values:
        if isinstance(value,(int,float)) and not isinstance(value,bool):result.append(float(value))
        elif isinstance(value,(tuple,list)):result.extend(numbers(value))
    return result


def differences(left,right):
    left=numbers(left);right=numbers(right)
    aligned=len(left)==len(right)
    finite=all(math.isfinite(x) for x in left+right)
    result=dict(reference_count=len(left),candidate_count=len(right),same_count=aligned,all_finite=finite)
    if not aligned or not finite or not left:
        result.update(max_abs=None,mean_abs=None,relative_l2=None)
        return result
    delta=[b-a for a,b in zip(left,right)];norm=math.sqrt(sum(a*a for a in left))
    result.update(max_abs=max(abs(d) for d in delta),mean_abs=statistics.mean(abs(d) for d in delta),
                  relative_l2=math.sqrt(sum(d*d for d in delta))/max(norm,1e-30),
                  reference_mean=statistics.mean(left),candidate_mean=statistics.mean(right))
    return result


def metric(row,name):
    return [value for timing in row.get('timings',[]) for value in numbers([timing.get('metrics',{}).get(name,[])])]


def forwards(row):return [value for timing in row.get('timings',[]) for value in timing.get('train_forwards',[])]


def parameter_difference(left,right,field):
    a={row['name']:row for row in left.get(field,[])};b={row['name']:row for row in right.get(field,[])}
    names=sorted(set(a)&set(b));same=set(a)==set(b) and bool(a)
    shape_equal=all((a[n]['shape'],a[n]['dtype'],a[n]['elements'])==
                    (b[n]['shape'],b[n]['dtype'],b[n]['elements']) for n in names)
    comparison=differences([a[n].get('samples',[]) for n in names],[b[n].get('samples',[]) for n in names])
    return dict(same_parameter_names=same,same_layout=shape_equal,matched_local_shards=len(names),
                sampled_hashes_equal=sum(a[n].get('sample_sha256')==b[n].get('sample_sha256') for n in names),
                numerical_samples=comparison,scope='at most64 reported elements per local parameter shard; not full parameters')


def adam_difference(left,right,field):
    a={row['index']:row for row in left.get(field,[])};b={row['index']:row for row in right.get(field,[])}
    ids=sorted(set(a)&set(b));same=set(a)==set(b) and bool(a);hash_equal=hash_count=0
    va=[];vb=[];state_keys_equal=True
    for index in ids:
        sa,sb=a[index].get('states',{}),b[index].get('states',{})
        state_keys_equal &= sa.keys()==sb.keys()
        for key in sorted(sa.keys()&sb.keys()):
            x,y=sa[key],sb[key]
            if isinstance(x,dict) and isinstance(y,dict):
                if 'sample_sha256' in x and 'sample_sha256' in y:
                    hash_count+=1;hash_equal+=int(x['sample_sha256']==y['sample_sha256'])
                # Current reports carry hashes for Adam samples. If future
                # reports also carry actual values, compare only those values.
                if isinstance(x.get('samples'),list) and isinstance(y.get('samples'),list):
                    va.extend(x['samples']);vb.extend(y['samples'])
    return dict(same_state_indices=same,same_state_keys=state_keys_equal,
                step_counters=differences([a[i].get('step') for i in ids],[b[i].get('step') for i in ids]),
                sampled_state_hashes=dict(compared=hash_count,equal=hash_equal,different=hash_count-hash_equal),
                numerical_moment_samples=differences(va,vb) if va or vb else None,
                numerical_moment_difference_available=bool(va or vb),
                scope='reported Adam step counters and64-element sample hashes per state tensor; '
                      'hash mismatch supplies no numerical error magnitude; not full Adam equality')


def completed_work(row):
    timings=row.get('timings',[]);fwd=forwards(row)
    return dict(twenty_adam_writes=row.get('actual_optimizer_writes')==20,
                twenty_adam_calls=row.get('actual_optimizer_calls')==20,
                twenty_minibatches=[t.get('index') for t in timings]==list(range(20)),
                eighty_training_forwards=len(fwd)==80 and row.get('actual_training_forwards')==80,
                original_local640_rows=len(row.get('input_data',{}).get('local_rows',[]))==640)


def compare(legacy,lean):
    ids=sorted(set(legacy['ranks'])&set(lean['ranks']));rows=[]
    same_four=ids==[0,1,2,3] and sorted(legacy['ranks'])==ids and sorted(lean['ranks'])==ids
    for rank in ids:
        a,b=legacy['ranks'][rank],lean['ranks'][rank];fa,fb=forwards(a),forwards(b)
        work_a,work_b=completed_work(a),completed_work(b)
        data_a,data_b=a.get('input_data',{}),b.get('input_data',{})
        same_data=data_a==data_b
        same_tokens=[f.get('authored_tokens') for f in fa]==[f.get('authored_tokens') for f in fb]
        wall_a,wall_b=a.get('elapsed_seconds'),b.get('elapsed_seconds')
        rows.append(dict(rank=rank,legacy_work=work_a,leancompact_work=work_b,
            same_checkpoint=a.get('checkpoint')==b.get('checkpoint'),same_original_data_identity=same_data,
            same_forward_authored_token_counts=same_tokens,
            wall_seconds=dict(legacy=wall_a,leancompact=wall_b,
                legacy_over_leancompact=wall_a/wall_b if wall_a and wall_b else None),
            actual_monitor_forwards=dict(legacy=a.get('actual_monitor_forwards'),leancompact=b.get('actual_monitor_forwards')),
            lp_samples=differences([f.get('samples',[]) for f in fa],[f.get('samples',[]) for f in fb]),
            lp_means=differences([f.get('mean') for f in fa],[f.get('mean') for f in fb]),
            policy_loss=differences(metric(a,'actor/pg_loss'),metric(b,'actor/pg_loss')),
            gradient_norm=differences(metric(a,'actor/grad_norm'),metric(b,'actor/grad_norm')),
            clip_fraction=differences(metric(a,'actor/pg_clipfrac'),metric(b,'actor/pg_clipfrac')),
            ppo_kl=differences(metric(a,'actor/ppo_kl'),metric(b,'actor/ppo_kl')),
            initial_parameter_samples=parameter_difference(a,b,'initial_parameter_samples'),
            final_parameter_samples=parameter_difference(a,b,'final_parameter_samples'),
            initial_adam_samples=adam_difference(a.get('restored_actor_state',{}),b.get('restored_actor_state',{}),'actual_adam_samples'),
            final_adam_samples=adam_difference(a,b,'final_adam_samples'),
            scheduler_before_equal=a.get('scheduler_before')==b.get('scheduler_before'),
            scheduler_after_equal=a.get('scheduler_after')==b.get('scheduler_after'),
            global_cpu_rng_equal=a.get('restored_actor_state',{}).get('global_torch_cpu_rng_sha256')==
                                 b.get('restored_actor_state',{}).get('global_torch_cpu_rng_sha256'),
            global_cuda_rng_equal=a.get('restored_actor_state',{}).get('global_torch_cuda_rng_sha256')==
                                  b.get('restored_actor_state',{}).get('global_torch_cuda_rng_sha256')))
    matched=bool(same_four and all(all(r['legacy_work'].values()) and all(r['leancompact_work'].values()) and
        r['same_checkpoint'] and r['same_original_data_identity'] and r['same_forward_authored_token_counts'] for r in rows))
    max_a=max((r['wall_seconds']['legacy'] for r in rows if r['wall_seconds']['legacy'] is not None),default=None)
    max_b=max((r['wall_seconds']['leancompact'] for r in rows if r['wall_seconds']['leancompact'] is not None),default=None)
    return dict(format='native4_fixed_work_fastpath_comparison_v1',same_four_ranks=same_four,
        matching_twenty_update_work_proven_by_reports=matched,
        legacy_root=legacy['root'],leancompact_root=lean['root'],
        fixed_work_wall_seconds=dict(legacy_slowest_rank=max_a,leancompact_slowest_rank=max_b),
        fixed_work_wall_speedup=max_a/max_b if matched and max_a and max_b else None,
        ranks=rows,whole_outer_speedup_proven=False,full_parameter_numerical_equivalence_proven=False,
        full_optimizer_numerical_equivalence_proven=False,production_reward_equivalence_proven=False,
        scope='two independent native4 actor jobs, identical fixed saved trajectories and diagnostic EM advantages; '
              'numerical comparisons cover only reported LP, loss, norm and parameter/Adam samples; '
              'no fresh rollout, QA or full training-engine acceptance')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy',required=True);parser.add_argument('--leancompact',required=True)
    parser.add_argument('--output',help='Optional small comparison JSON; source reports remain untouched')
    args=parser.parse_args(argv);report=compare(reports(args.legacy),reports(args.leancompact))
    content=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    print(content,end='')


if __name__=='__main__':main()
