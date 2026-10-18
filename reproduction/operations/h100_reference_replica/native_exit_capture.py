"""Collect actual kernel exits of explicitly bound, already running leaders.

No ptrace, signal, training edit, model load, GPU use, or global kernel change.
Pipeline markers never substitute for an actual process-exit event.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import time

from kernel_exit_events import Connector


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def once(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    once(temp,value);os.replace(temp,path)
def birth(pid):
    stat=Path('/proc',str(pid),'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
    return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=Path('/proc',str(pid)).stat().st_uid,
        state=fields[0],command=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode())
def same(actual,expected):return all(actual[key]==expected[key]for key in ('pid','start_ticks','sid','uid'))


def prerequisites(root):
    manifest=read(root/'manifest.json');qualification=read(manifest['qualification_result'])
    if qualification['status']!='OWNED_CPU_KERNEL_EXIT_CAPTURE_0_AND_7_PASS':raise ValueError('Kernel exit ABI not independently qualified')
    if qualification['host']!=socket.gethostname()or qualification['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('Qualification is for another kernel host/boot')
    decoder=Path(__file__).with_name('kernel_exit_events.py')
    if sha(decoder)!=qualification['source_sha256']:raise ValueError('Decoder differs from the actually qualified implementation')
    return manifest


def run(root):
    root=Path(root);lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=prerequisites(root);pidfds={};proofs={};pending={};armed={}
    common=dict(host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        manifest_sha256=sha(root/'manifest.json'),collector_sha256=sha(__file__),
        collector_birth=birth(os.getpid()),training_modified=False,training_signals_sent=False,
        gpu_or_model_test=False,global_kernel_policy_changed=False)
    def heartbeat():
        atomic(root/'status.json',dict(common,status='PASSIVE_KERNEL_EXIT_CAPTURE_ARMED',
            bound_targets=armed,captured_arms=sorted(proofs),pending_pidfd_confirmation=sorted(pending),
            checked_unix=time.time()))
    try:
        with Connector()as connector:
            # Subscription precedes birth checks, avoiding the verify/subscribe race.
            for arm,target in manifest['targets'].items():
                expected=target['leader'];pid=expected['pid'];actual=birth(pid)
                receipt=Path(target['base'])/'preparation/submit-receipt.json';record=read(receipt)
                if not same(actual,expected)or not same(record['leader'],expected):raise ValueError('Actual leader birth differs: '+arm)
                if record['run_id']!=target['run_id']or record['config_sha256']!=target['config_sha256']:
                    raise ValueError('Submitted run/config differs: '+arm)
                if target['base']not in actual['command']or 'reproduction.prefix_selection200.launch'not in actual['command']:
                    raise ValueError('Actual leader command differs: '+arm)
                pidfds[arm]=os.pidfd_open(pid)
                if not same(birth(pid),expected):raise ValueError('PID reused during pidfd binding')
                armed[arm]=dict(target,actual_birth=actual,submit_receipt=str(receipt),submit_receipt_sha256=sha(receipt),
                    pid_namespace=os.readlink('/proc/'+str(pid)+'/ns/pid'))
                if armed[arm]['pid_namespace']!=os.readlink('/proc/self/ns/pid'):raise ValueError('Leader/collector PID namespace differs')
            once(root/'ARMED.json',dict(common,bound_targets=armed,armed_unix=time.time(),actual_exit_still_pending=True))
            heartbeat();last=time.monotonic()
            while len(proofs)<len(armed):
                try:events=connector.receive(1)
                except socket.timeout:events=[]
                for event in events:
                    if event['kind']!='exit'or event['pid']!=event['tgid']:continue
                    for arm,target in armed.items():
                        if event['pid']==target['leader']['pid']and arm not in proofs:
                            if event['kernel_monotonic_ns']<target['leader']['start_ticks']*1e9/os.sysconf('SC_CLK_TCK'):
                                raise ValueError('Exit event predates the bound process birth')
                            pending[arm]=event
                for arm,target in armed.items():
                    if arm in proofs:continue
                    if select.select([pidfds[arm]],[],[],0)[0]:
                        if arm not in pending:raise RuntimeError('Bound leader exited without a captured kernel event; exit0 unknown')
                        event=pending.pop(arm)
                        proof=dict(common,target=target,event=event,actual_bound_pidfd_exit_confirmed=True,
                            status='ACTUAL_BOUND_NATIVE_EXIT_CAPTURED',captured_unix=time.time(),
                            actual_native_exit_zero=event['exit_code']==0 and event['termination_signal']is None,
                            final_qa_checkpoints_retention_or_owned_cleanup_verified=False)
                        once(root/(arm+'-actual-exit.json'),proof);proofs[arm]=proof;heartbeat()
                if time.monotonic()-last>=60:heartbeat();last=time.monotonic()
            once(root/'RESULT.json',dict(common,status='ALL_BOUND_NATIVE_KERNEL_EXITS_CAPTURED',proofs=proofs,
                all_actual_native_exit_zero=all(p['actual_native_exit_zero']for p in proofs.values()),
                whole_goal_complete=False,completed_unix=time.time()))
    except BaseException as error:
        if not(root/'RESULT.json').exists():once(root/'RESULT.json',dict(common,status='EXIT_CAPTURE_UNAVAILABLE_NATIVE_STATUS_NOT_INFERRED',
            error=repr(error),captured_arms=sorted(proofs),whole_goal_complete=False,training_modified=False))
        raise
    finally:
        for fd in pidfds.values():os.close(fd)


def submit(root):
    root=Path(root);prerequisites(root)
    if(root/'submit-intent.json').exists():return status(root)
    once(root/'submit-intent.json',dict(manifest_sha256=sha(root/'manifest.json'),collector_sha256=sha(__file__),
        created_unix=time.time(),training_modified=False))
    with(root/'collector.log').open('ab',buffering=0)as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    once(root/'submit-receipt.json',dict(pid=child.pid,birth=birth(child.pid),submitted_unix=time.time(),training_modified=False))
    return status(root)


def status(root):
    root=Path(root);value=dict(root=str(root),training_modified=False)
    for name in ('submit-intent.json','submit-receipt.json','ARMED.json','status.json','RESULT.json','A-actual-exit.json','B-actual-exit.json'):
        path=root/name
        if path.exists():value[name]=dict(path=str(path),sha256=sha(path),value=read(path))
    if 'submit-receipt.json'in value:
        expected=value['submit-receipt.json']['value']['birth']
        try:actual=birth(expected['pid']);value['collector_same_birth']=same(actual,expected);value['collector_state']=actual['state']
        except FileNotFoundError:value['collector_same_birth']=False
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('submit','run','status'));parser.add_argument('--root',required=True)
    args=parser.parse_args()
    if args.action=='run':run(args.root)
    else:print(json.dumps(globals()[args.action](Path(args.root))))
