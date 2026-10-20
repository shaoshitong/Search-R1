"""Read-only initial whole-batch follow of the two extra source5 consumers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from .source_sublease_control import ROOT, GROUPS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hours',type=float,default=4)
    p.add_argument('--interval',type=int,default=60)
    p.add_argument('--own-batches',type=int,default=1)
    a=p.parse_args();assert a.interval>=30 and a.own_batches>=1
    latest=ROOT/'work/qwen7b-source5-extra-first-batches-latest-small.json'
    until=time.monotonic()+a.hours*3600;last=None
    while time.monotonic()<until:
        rows={}
        for group in GROUPS:
            try:
                r=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_resample.source_sublease_control',
                    'status','--group',group],text=True,capture_output=True,timeout=110)
                assert r.returncode==0,r.stderr[-1000:]
                outer=json.loads(r.stdout);v=json.loads(outer['stdout']);s=v.get('sampler')or{};progress=s.get('progress')or{}
                model=s.get('model_initialized')or[]
                phases=[dict(batch=x['batch'],phase=x['phase'],completed=x['metadata'].get('completed_trajectories'),
                    actual_sampled_tokens=x['metadata'].get('actual_sampled_tokens'),utc=x['metadata'].get('utc'))for x in s.get('active_phases',[])]
                previous=Path(progress.get('latest_input','')).stem
                next_started=bool(previous and any(x['batch']!=previous and x['completed']for x in phases))
                rows[group]=dict(epoch=v['epoch'],unit=v['unit'],observed=v['observed'],
                    model_ready=bool(model),actual_kv_blocks=model[-1]['actual_num_gpu_blocks']if model else None,
                    own_complete_batches=progress.get('completed_own_batches',0),next_input_started=next_started,
                    phases=phases,failure=s.get('failure'),native_complete=s.get('complete'),actual=outer['receipt'])
            except Exception as error:
                rows[group]=dict(status='OBSERVATION_CHANNEL_ERROR_NOT_NATIVE_EXIT',error=str(error)[-1000:],epoch=time.time())
        value=dict(groups=rows,goal_complete=False,epoch=time.time())
        latest.write_text(json.dumps(value,indent=2)+'\n')
        change=[(g,x.get('own_complete_batches'),x.get('phases'),x.get('failure'),x.get('status'))for g,x in rows.items()]
        if change!=last:print(json.dumps(value),flush=True);last=change
        if any(x.get('failure')for x in rows.values()):print('ROOT_ATTENTION_ACTUAL_EXTRA_SOURCE_SAMPLER_FAILURE',flush=True);return
        if all(x.get('own_complete_batches',0)>=a.own_batches and x.get('next_input_started')for x in rows.values()):
            print('ROOT_ATTENTION_BOTH_EXTRA_SOURCE_CONSUMERS_REAL_WHOLE_BATCH_AND_NEXT_FLOW',flush=True);return
        if any(x.get('native_complete')for x in rows.values()):print('ROOT_ATTENTION_EXTRA_SOURCE_ENDED_READ_ACTUAL_REASON',flush=True);return
        time.sleep(a.interval)
    print('ROOT_ATTENTION_SOURCE_FOLLOW_WINDOW_ENDED_GOAL_INCOMPLETE',flush=True)


if __name__=='__main__':main()
