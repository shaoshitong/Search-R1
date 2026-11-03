"""Sequential high-utilization follow-up; isolated from the baseline job."""
import json, time, os
from driver import run_case, ROOT, RESULTS

def main():
    for util in [.98,.99,1.0]:
        label='bf16-u'+str(round(util*1000)).zfill(3)
        profile=run_case(label+'-profile',util,40960,[],limit=240)
        if profile['pass'] and profile['capacity_tokens']:
            cap=profile['capacity_tokens']//16*16
            result=run_case(label+'-exact-capacity',util,cap,[cap-1],limit=600)
            if not result['pass']:
                # Profiling and startup can vary slightly; preserve failed run.
                run_case(label+'-near-capacity',util,cap-1024,[cap-1025],limit=600)
    req=[dict(case=c['name'],**e) for c in RESULTS if c['pass'] for e in c['events'] if e['event']=='request_pass']
    if not req:
        raise RuntimeError('No high-utilization request passed; initialization failures are not a capacity result')
    (ROOT/'complete.json').write_text(json.dumps(dict(completed_at=time.time(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),requests=req,
        largest_validated_total_tokens=max([x['total_tokens'] for x in req],default=0),
        cases=[{k:c[k] for k in ['name','pass','exit_code','capacity_tokens','max_model_len']} for c in RESULTS]),indent=2))
    print('HIGH_UTILIZATION_COMPLETE',flush=True)

if __name__=='__main__':main()
