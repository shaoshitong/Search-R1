"""Foreground read-only followup of the existing 731063 recovery through outer10/11.

It never starts or changes training. Transport/terminal/real error observations
return to the caller for diagnosis rather than causing an automatic submission.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import time

IDENTITY=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',
              pid_namespace='pid:[4026531836]')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bridge-source',required=True);p.add_argument('--control-work',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=10800)
    a=p.parse_args()
    if not 1<=a.max_seconds<=14400:raise ValueError('Bounded foreground observation only')
    repo=Path(__file__).resolve().parents[2]
    binding=json.loads((repo/'reproduction/easy2560_joint4/code_backup/recovery5-731063-20261018/current-readonly-followup-commands.json').read_text())
    assert binding['job']=='731063' and binding['source_commit']=='a61b48139386b346e4fef12511ece1ac0d09a8e3'
    spec=importlib.util.spec_from_file_location('account_bridge',a.bridge_source)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+a.max_seconds;previous=None;polls=0;transport_errors=0
    while time.monotonic()<deadline:
        polls+=1
        result=bridge.run('jxu957',binding['iteration_progress_command'],'',a.control_work,
            login_host='10.120.48.31',expected_host_identity=IDENTITY,observation_timeout=55)
        row=dict(checked_unix=time.time(),polls=polls,returncode=result.returncode)
        if result.returncode:
            transport_errors+=1
            row.update(status='TRANSPORT_UNKNOWN_NO_TRAINING_ACTION',stderr=result.stderr[-1800:])
            print(json.dumps(row),flush=True)
            (a.output_dir/('transport-'+str(int(time.time()))+'.json')).write_text(json.dumps(row,indent=2)+'\n')
            if transport_errors>=3:return
        else:
            transport_errors=0;d=json.loads(result.stdout)
            assert d['job']=='731063'
            completed=d.get('actual_completed') or []
            last=completed[-1].get('step',0) if completed else 0
            candidate=(d.get('candidate_chunks') or [{}])[-1]
            next_started=bool(re.search(r'epoch\s+0,\s+step\s+11(?:\D|$)',d.get('train_tail',''))) or candidate.get('step',0)>=11
            key=json.dumps(dict(queue=d['queue']['stdout'],stage=d['stage'],candidate=candidate,
                                completed=completed,ranks=d['ranks'],errors=d['error_lines']),sort_keys=True)
            if key!=previous:
                path=a.output_dir/('progress-'+str(int(time.time()))+'.json')
                path.write_text(json.dumps(d,indent=2)+'\n');previous=key
                print(json.dumps(dict(status='ACTUAL_PROGRESS_OBSERVATION',audit=str(path),polls=polls,
                    checked_unix=d['checked_unix'],queue=d['queue']['stdout'].strip(),
                    candidate=candidate,last_completed_outer=last,next11_started=next_started,
                    error_lines=d['error_lines'])),flush=True)
            if d['error_lines'] or '|RUNNING|' not in d['queue']['stdout']:
                print(json.dumps(dict(status='ATTENTION_REQUIRED_NO_AUTOMATIC_SUBMISSION',job='731063')),flush=True)
                return
            if last>=10 and next_started:
                print(json.dumps(dict(status='RECOVERY_FIRST_FIVE_COMPLETE_AND_NEXT11_STARTED',job='731063',
                    completed_outer=last,goal_complete=False,training_stop_requested=False)),flush=True)
                return
        time.sleep(min(45,max(0,deadline-time.monotonic())))
    print(json.dumps(dict(status='FOREGROUND_OBSERVATION_TIME_BOUND',goal_complete=False,training_unchanged=True)),flush=True)


if __name__=='__main__':main()

