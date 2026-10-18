"""Once control for actual A/B normal-savepoint CPU readback; no trainer edit.

Start never retries transport or a consumed intent. Status reads small receipts,
proofs and bounded logs only. A uses its frozen worker; progressive B uses a
versioned preparation-only reader for the completed stage's20/40/80 dimensions.
"""
import argparse
import base64
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from .follow_h10011_ab_normal_boundary import ROOT, PATH_LAYOUTS
from .readonly_management_route import MODES, lan5_argv, run_readonly


INNER = r'''
import base64,hashlib,json,os,subprocess,time
from pathlib import Path
ARMS=__ARMS__
OUTER=__OUTER__
ACTION=__ACTION__
PROGRESSIVE_WORKER=__PROGRESSIVE_WORKER__
result={}
def birth(receipt):
 p=Path('/proc/'+str(receipt['pid'])+'/stat')
 try:
  fields=p.read_text().rsplit(')',1)[1].split()
  return int(fields[19])==receipt['start_ticks'] and fields[0]!='Z'
 except FileNotFoundError:return False
for arm,entry in ARMS.items():
 b=Path(entry['base']);prep=b/'preparation'
 proof=prep/('ab-joint-%04d-independent-full-sha-cpu.json'%OUTER)
 receipt=prep/('ab-cp%d-cpu-control-receipt.json'%OUTER)
 intent=prep/('ab-cp%d-cpu-control-intent.json'%OUTER)
 worker_intent=proof.with_suffix('.intent.json')
 log=prep/('ab-cp%d-cpu-control.log'%OUTER)
 value={'outer':OUTER,'base':str(b),'run_id':entry['run_id']}
 if proof.exists():
  raw=proof.read_bytes();saved=json.loads(raw)
  value.update(status='EXISTING_CPU_PROOF_NO_REPLAY',proof=str(proof),
   proof_sha256=hashlib.sha256(raw).hexdigest(),result=saved)
 elif receipt.exists():
  saved=json.loads(receipt.read_text())
  value.update(status='EXISTING_RECEIPT_NO_REPLAY',receipt=saved,
   same_birth_now=birth(saved),proof_exists=False)
 elif intent.exists() or worker_intent.exists():
  value.update(status='EXISTING_INTENT_QUERY_NO_REPLAY',
   control_intent_exists=intent.exists(),worker_intent_exists=worker_intent.exists())
 elif not (b/'outers'/('outer-%06d'%OUTER)/'complete.json').exists():
  value['status']='PENDING_NO_COMMITTED_JOINT_NO_AUDIT'
 elif ACTION=='status':
  value['status']='COMMITTED_NOT_YET_CPU_AUDITED'
 else:
  config_path=prep/'launch-config.json';raw=config_path.read_bytes();config=json.loads(raw)
  if hashlib.sha256(raw).hexdigest()!=entry['config_sha256'] or config['run_id']!=entry['run_id']:
   raise RuntimeError('Actual launch binding differs; do not start a checkpoint reader')
  progressive=config.get('b_progressive',{}).get('mode')=='b_cp50_prefix20_40_80_two_stage_v1'
  prefix=(5 if OUTER<=50 else 20 if OUTER<=100 else 40 if OUTER<=150 else 80) if progressive else 5
  claim={'outer':OUTER,'run_id':entry['run_id'],'source_commit':entry['commit'],
   'config_sha256':entry['config_sha256'],'created_unix':time.time(),
   'cpu_only':True,'training_not_stopped':True,'transport_auto_retry':False,'checkpoint_prefix_tokens':prefix}
  env=dict(os.environ,PYTHONPATH=str(b/'repo'),CUDA_VISIBLE_DEVICES='',
   OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
   SEARCH_G1_B_PREFIX_TOKENS=str(prefix))
  command=['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-m',
   'reproduction.prefix_selection200.ab_checkpoint_once','--base',str(b),'--outer',str(OUTER)]
  if progressive:
   # A versioned CPU-only reader lives beside preparation evidence. The active
   # frozen repo is never overwritten, even when a later boundary changes width.
   payload=base64.b64decode(PROGRESSIVE_WORKER['base64'])
   checksum=hashlib.sha256(payload).hexdigest()
   if checksum!=PROGRESSIVE_WORKER['sha256']:raise ValueError('CPU worker payload SHA differs')
   worker=prep/'normal-savepoint-tools'/checksum/'ab_checkpoint_once.py'
   worker.parent.mkdir(parents=True,exist_ok=True)
   if not worker.exists():
    with worker.open('xb') as stream:stream.write(payload)
   if hashlib.sha256(worker.read_bytes()).hexdigest()!=checksum:raise ValueError('Staged CPU worker SHA differs')
   code='import importlib.util,json;spec=importlib.util.spec_from_file_location("reproduction.prefix_selection200._normal_checkpoint_control",'+repr(str(worker))+');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);print(json.dumps(m.inspect('+repr(str(b))+','+repr(OUTER)+')))'
   command=['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',code]
   claim.update(cpu_control_source=str(worker),cpu_control_source_sha256=checksum,frozen_repo_changed=False)
  with intent.open('x') as stream:json.dump(claim,stream)
  with log.open('ab') as stream:
   process=subprocess.Popen(command,env=env,cwd=b/'repo',stdout=stream,
    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  stat=Path('/proc/'+str(process.pid)+'/stat').read_text().rsplit(')',1)[1].split()
  saved=dict(claim,status='CPU_READBACK_STARTED_NOT_PASS',pid=process.pid,
   start_ticks=int(stat[19]),proof=str(proof),log=str(log),command=command)
  with receipt.open('x') as stream:json.dump(saved,stream)
  value.update(status=saved['status'],receipt=saved)
 if log.exists():
  with log.open('rb') as stream:
   stream.seek(max(0,log.stat().st_size-3500));value['log_tail']=stream.read().decode(errors='replace')
 result[arm]=value
print(json.dumps(result))
'''


def remote_inner(arms, outer, action):
    if outer not in (50, 100, 150, 200) or action not in ('start', 'status'):
        raise ValueError('Only actual normal savepoints and explicit actions are supported')
    worker=Path(__file__).resolve().parents[2]/'prefix_selection200/ab_checkpoint_once.py'
    payload=worker.read_bytes()
    substitutions={'__ARMS__':repr(arms),'__OUTER__':repr(outer),'__ACTION__':repr(action),
        '__PROGRESSIVE_WORKER__':repr(dict(base64=base64.b64encode(payload).decode(),sha256=hashlib.sha256(payload).hexdigest()))}
    # Replace template tokens in one pass: inserted paths/source are never
    # interpreted as further placeholders.
    import re
    return re.sub('|'.join(map(re.escape,substitutions)),lambda match:substitutions[match.group()],INNER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status'))
    parser.add_argument('--outer', type=int, required=True, choices=(50, 100, 150, 200))
    parser.add_argument('--arm', choices=('A', 'B', 'both'), default='both')
    parser.add_argument('--transport', choices=MODES, default='retained18')
    parser.add_argument('--path-layout', choices=tuple(PATH_LAYOUTS), default='source8')
    args = parser.parse_args()
    if args.action == 'start' and args.transport == 'retained18_fallback_lan5':
        parser.error('Once start cannot automatically retry transport; query status after an uncertain outcome')
    arms = json.loads((ROOT/'work/prefix5-ab-assigned-arms.json').read_text())
    if args.arm != 'both':
        arms = {args.arm: arms[args.arm]}
    if not all(entry['node'] == 'h100-11' for entry in arms.values()):
        raise ValueError('This control is scoped to the current qualified h100-11 placement')
    inner = remote_inner(arms, args.outer, args.action)
    script = ('import subprocess,os\np=subprocess.run('+repr([
        'unshare', '--mount', '--propagation', 'private', 'bash', PATH_LAYOUTS[args.path_layout],
        '/usr/bin/python3', '-c', inner])+
        ',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),'
        'text=True,capture_output=True,timeout=40)\n'
        'assert p.returncode==0,p.stderr\nprint(p.stdout)')
    spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctl)
    tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    evidence = ROOT/('work/prefix5-ab-normal-cp%d-%s-%s.json' % (args.outer, args.action, tag))
    try:
        if args.action == 'status':
            response, route = run_readonly(ctl, 'h100-11', '/usr/bin/python3 -',
                input_text=script, timeout=60, mode=args.transport)
        else:
            argv = (lan5_argv(ctl, 'h100-11', '/usr/bin/python3 -') if args.transport == 'lan5'
                    else ctl.route('h100-11', '/usr/bin/python3 -'))
            response = subprocess.run(argv, input=script, text=True, capture_output=True, timeout=60)
            route = dict(route=args.transport, auto_retry=False)
    except subprocess.TimeoutExpired:
        evidence.write_text(json.dumps(dict(status='TRANSPORT_TIMEOUT_QUERY_SAME_ONCE_STATE_DO_NOT_REPLAY',
            action=args.action, outer=args.outer, training_modified=False), indent=2)+'\n')
        raise
    value = dict(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr,
                 transport=route, action=args.action, outer=args.outer, training_modified=False,
                 observation_namespace=dict(path_layout=args.path_layout,
                     wrapper=PATH_LAYOUTS[args.path_layout],live_training_changed=False))
    evidence.write_text(json.dumps(value, indent=2)+'\n')
    if response.returncode:
        raise RuntimeError(f'Control outcome saved at {evidence}; query once state, never blind replay: {response.stderr}')
    print(json.dumps(dict(evidence=str(evidence), arms=json.loads(response.stdout), transport=route)))


if __name__ == '__main__':
    main()
