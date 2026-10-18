"""Foreground read-only followup of the existing 731063 recovery.

It never starts or changes training. Transport/terminal/real error observations
return to the caller for diagnosis rather than causing an automatic submission.
By default it observes recovery through outer10 and the start of outer11. An
explicit completed-outer boundary only stops this local observer.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

IDENTITY=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',
              pid_namespace='pid:[4026531836]')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bridge-source',required=True);p.add_argument('--control-work',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--command-bindings',type=Path,
                   help='Explicit published read-only job/commit/command binding; defaults to historical731063')
    p.add_argument('--max-seconds',type=int,default=10800)
    p.add_argument('--stop-after-completed-outer',type=int,
                   help='Stop only this observer at an actual completed outer1..200; no next outer required')
    a=p.parse_args()
    if not 1<=a.max_seconds<=86400:raise ValueError('Bounded foreground observation only')
    if a.stop_after_completed_outer is not None and not 1<=a.stop_after_completed_outer<=200:
        raise ValueError('Completed observation boundary must be outer1..200')
    repo=Path(__file__).resolve().parents[2]
    binding_path=a.command_bindings or repo/'reproduction/easy2560_joint4/code_backup/recovery5-731063-20261018/current-readonly-followup-commands.json'
    binding=json.loads(binding_path.read_text());job=binding['job']
    assert isinstance(job,str) and re.fullmatch('[1-9][0-9]*',job)
    assert re.fullmatch('[0-9a-f]{40}',binding['source_commit'])
    spec=importlib.util.spec_from_file_location('account_bridge',a.bridge_source)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+a.max_seconds;previous=None;polls=0;transport_errors=0
    while time.monotonic()<deadline:
        polls+=1
        try:
            result=bridge.run('jxu957',binding['iteration_progress_command'],'',a.control_work,
                login_host='10.120.48.31',expected_host_identity=IDENTITY,observation_timeout=55)
        except (subprocess.TimeoutExpired,TimeoutError):
            result=subprocess.CompletedProcess([],124,'','Read-only observation timed out')
        except Exception as error:
            print(json.dumps(dict(status='CPU_OBSERVER_WRAPPER_FAILED_ATTENTION_REQUIRED',job=job,
                checked_unix=time.time(),polls=polls,error=repr(error),goal_complete=False,
                training_stop_requested=False,no_automatic_submission=True)),flush=True)
            return
        row=dict(checked_unix=time.time(),polls=polls,returncode=result.returncode)
        if result.returncode:
            # An SSH wrapper's own timeout can surface as a Python traceback.
            transport_unknown=result.returncode in (124,255,86) or 'TimeoutExpired' in result.stderr
            wrapper_error=('Traceback (most recent call last)' in result.stderr or 'SyntaxError:' in result.stderr)
            if not transport_unknown and wrapper_error:
                row.update(status='CPU_READONLY_WRAPPER_FAILED_ATTENTION_REQUIRED',stderr=result.stderr[-1800:],
                           goal_complete=False,training_stop_requested=False,no_automatic_submission=True)
                print(json.dumps(row),flush=True)
                (a.output_dir/('wrapper-failure-'+str(int(time.time()))+'.json')).write_text(json.dumps(row,indent=2)+'\n')
                return
            transport_errors+=1
            row.update(status='TRANSPORT_UNKNOWN_NO_TRAINING_ACTION',stderr=result.stderr[-1800:])
            print(json.dumps(row),flush=True)
            (a.output_dir/('transport-'+str(int(time.time()))+'.json')).write_text(json.dumps(row,indent=2)+'\n')
            if transport_errors>=3:return
        else:
            try:
                d=json.loads(result.stdout)
                assert d['job']==job
            except (ValueError,KeyError,TypeError,AssertionError) as error:
                row.update(status='CPU_READONLY_WRAPPER_OUTPUT_INVALID_ATTENTION_REQUIRED',error=repr(error),
                           stderr=result.stderr[-1800:],goal_complete=False,training_stop_requested=False,
                           no_automatic_submission=True)
                print(json.dumps(row),flush=True)
                (a.output_dir/('wrapper-failure-'+str(int(time.time()))+'.json')).write_text(json.dumps(row,indent=2)+'\n')
                return
            transport_errors=0
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
            stage=d['stage'].get('stage') if isinstance(d['stage'],dict) else d['stage']
            if d['error_lines'] or stage in ('FAILED','OWNED_CLEANUP_FAILED'):
                print(json.dumps(dict(status='ATTENTION_REQUIRED_NO_AUTOMATIC_SUBMISSION',job=job)),flush=True)
                return
            if '|RUNNING|' not in d['queue']['stdout']:
                print(json.dumps(dict(status='ATTENTION_REQUIRED_NO_AUTOMATIC_SUBMISSION',job=job)),flush=True)
                return
            if a.stop_after_completed_outer is not None and last>=a.stop_after_completed_outer:
                print(json.dumps(dict(status='ACTUAL_COMPLETED_OUTER_BOUNDARY_OBSERVED',job=job,
                    requested_completed_outer=a.stop_after_completed_outer,completed_outer=last,
                    checked_unix=d['checked_unix'],goal_complete=False,acceptance_claimed=False,
                    training_stop_requested=False)),flush=True)
                return
            if a.stop_after_completed_outer is None and last>=10 and next_started:
                print(json.dumps(dict(status='RECOVERY_FIRST_FIVE_COMPLETE_AND_NEXT11_STARTED',job=job,
                    completed_outer=last,goal_complete=False,training_stop_requested=False)),flush=True)
                return
        time.sleep(min(45,max(0,deadline-time.monotonic())))
    print(json.dumps(dict(status='FOREGROUND_OBSERVATION_TIME_BOUND',goal_complete=False,training_unchanged=True)),flush=True)


if __name__=='__main__':main()

