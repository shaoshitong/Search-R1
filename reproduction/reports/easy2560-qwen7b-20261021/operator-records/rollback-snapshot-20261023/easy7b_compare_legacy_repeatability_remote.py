import json,sys,time
from pathlib import Path
probe=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022')
sys.path.insert(0,str(probe/'source/d87d3b54c505b581250b152bccef0f01e9a5f071'))
from reproduction.operations.easy7b_20261021.compare_native4_fastpath import reports,compare
old=reports(probe/'diagnostics/native4-legacy-r1')
new=reports(probe/'diagnostics/native4-rollback-snapshot-r1-20261023')
mapping={'legacy_work':'old_legacy_work','leancompact_work':'new_legacy_work','legacy_root':'old_legacy_root','leancompact_root':'new_legacy_root','legacy_slowest_rank':'old_legacy_slowest_rank','leancompact_slowest_rank':'new_legacy_slowest_rank','legacy':'old_legacy','leancompact':'new_legacy','legacy_over_leancompact':'old_over_new','fixed_work_wall_speedup':'old_over_new_wall_ratio'}
def rename(value):
 if isinstance(value,dict):return {mapping.get(k,k):rename(v)for k,v in value.items()}
 if isinstance(value,list):return [rename(v)for v in value]
 return value
result=rename(compare(old,new))
result.update(format='native4_same_legacy_repeatability_samples_v1',epoch=time.time(),old_jobs=sorted({str(r.get('slurm_job_id'))for r in old['ranks'].values()}),new_jobs=sorted({str(r.get('slurm_job_id'))for r in new['ranks'].values()}),old_hosts=sorted({str(r.get('hostname'))for r in old['ranks'].values()}),new_hosts=sorted({str(r.get('hostname'))for r in new['ranks'].values()}),new_only_instrumentation='Original snapshot constructor wrapped with separately timed prior CUDA sync; original clone/forward/loss/Adam unchanged.',scope='Two independent legacy native4 CP50/selection51 fixed-work runs, one with constructor timing instrumentation. Numerical reported samples only; node/instrumentation variability is not a full-state or causal hardware comparison.',production_modified=False,new_gpu_work_started=False,source_reports_modified=False,model_or_tensor_loaded=False)
result['actual_write_growth']={str(rank):dict(before=r['restored_actor_state']['actual_optimizer_writes'],after=r['timings'][-1]['optimizer_writes'],actual_calls=r['actual_optimizer_calls'],actual_writes=r['actual_optimizer_writes'])for rank,r in new['ranks'].items()}
result['new_over_old_wall_change_fraction']=result['fixed_work_wall_seconds']['new_legacy_slowest_rank']/result['fixed_work_wall_seconds']['old_legacy_slowest_rank']-1
path=Path(new['root'])/'legacy-repeatability-vs-old.json'
with path.open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(remote_summary_path=str(path),summary=result),sort_keys=True))
