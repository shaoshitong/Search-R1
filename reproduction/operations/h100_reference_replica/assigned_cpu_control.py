"""Dispatch once CPU-only acceptance to each currently assigned native node.

No training launch, signal, GPU forward, frozen-repo edit or mutation retry.
Status may use the already-qualified read-only fallback on h100-11 only.
"""
import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
from .follow_h10011_ab_normal_boundary import ROOT
from .readonly_management_route import run_readonly,lan5_argv


def namespace_command(entry,inner):
    node=entry['node'];wrapper=entry['runtime_wrapper_path']
    if node not in ('h100-5','h100-11')or not wrapper.startswith('/root/.local/share/search-g1/source8-fault-recovery/'):
        raise ValueError('Require the actual qualified retained node/wrapper')
    command=['unshare','--mount','--propagation','private','bash',wrapper]
    if entry.get('runtime_wrapper_node_argument'):command.append(node)
    return command+['/usr/bin/python3','-c',inner]


def outer_script(entry,inner):
    env=dict(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
             **entry.get('required_process_environment',{}))
    return ('import subprocess,os\np=subprocess.run('+repr(namespace_command(entry,inner))+
            ',env=dict(os.environ,**'+repr(env)+'),text=True,capture_output=True,timeout=40)\n'
            'assert p.returncode==0,p.stderr\nprint(p.stdout)')


def execute(arms,build_inner,action,transport,label):
    if action not in ('start','status')or(action=='start'and transport=='retained18_fallback_lan5'):
        raise ValueError('Once start must choose a single transport; query after uncertainty')
    # Validate the entire dispatch before starting either independent CPU worker.
    scripts={arm:outer_script(entry,build_inner({arm:entry}))for arm,entry in arms.items()}
    if transport=='lan5'and any(e['node']!='h100-11'for e in arms.values()):
        raise ValueError('LAN5 fallback is qualified for h100-11 only; select that arm explicitly')
    spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    values={};evidence=[]
    for arm,entry in arms.items():
        node=entry['node'];tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path=ROOT/('work/prefix5-ab-'+label+'-'+arm+'-'+action+'-'+tag+'.json')
        try:
            if action=='status':
                mode='retained18'if node=='h100-5'and transport=='retained18_fallback_lan5'else transport
                response,route=run_readonly(ctl,node,'/usr/bin/python3 -',input_text=scripts[arm],timeout=60,mode=mode)
            else:
                argv=lan5_argv(ctl,node,'/usr/bin/python3 -')if transport=='lan5'else ctl.route(node,'/usr/bin/python3 -')
                response=subprocess.run(argv,input=scripts[arm],text=True,capture_output=True,timeout=60)
                route=dict(route=transport,auto_retry=False)
        except subprocess.TimeoutExpired:
            path.write_text(json.dumps(dict(status='TIMEOUT_QUERY_SAME_ONCE_STATE_DO_NOT_REPLAY',arm=arm,node=node,
                action=action,training_modified=False))+'\n');raise
        envelope=dict(returncode=response.returncode,stdout=response.stdout,stderr=response.stderr,arm=arm,node=node,
            transport=route,training_modified=False,frozen_repo_modified=False)
        path.write_text(json.dumps(envelope,indent=2)+'\n');evidence.append(str(path))
        if response.returncode:raise RuntimeError(str(path)+' '+response.stderr[-2500:])
        measured=json.loads(response.stdout)
        if set(measured)!={arm}:raise ValueError('CPU control response belongs to another arm')
        values.update(measured)
    return dict(evidence=evidence,arms=values,training_modified=False)
