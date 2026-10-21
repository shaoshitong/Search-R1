#!/usr/bin/env python3
"""Prepare/execute exact captured source5 retire steps; default is LOCAL preview.

Existing rollover_terminal is an observer, not an equivalent stop operator.
Execute extras, inspect their actual terminal states, then upstream, then root.
This never stops the namespace-pinning terminal collector or edits old outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = '/Users/shaoshitong/Documents/Search-G1/qwen7b-h100-assets-20261020'
BASE = '/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
PHYSICAL = '/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
DONE10_SHA = '596d7c0e44345e801aba3580da8aeb40958e08837da35bbf1398cc6210e91aca'

INNER = r'''
import fcntl, hashlib, importlib, json, os, socket, subprocess, sys, time
from pathlib import Path
assert os.geteuid()==0 and socket.gethostname().split('.')[0]=='h100-5'
BASE=Path(C['base']);PHYSICAL=Path(C['physical'])
def visible(raw):
 p=Path(raw)
 if p.is_relative_to(BASE):suffix=p.relative_to(BASE)
 elif p.is_relative_to(PHYSICAL):suffix=p.relative_to(PHYSICAL)
 else:raise AssertionError('Path outside exact retained NAS10 base')
 for candidate in (BASE/suffix,PHYSICAL/suffix):
  if candidate.exists():return candidate
 raise FileNotFoundError(str(raw))
def read(raw):return json.loads(visible(raw).read_text())
def sha(raw):return hashlib.sha256(visible(raw).read_bytes()).hexdigest()
assert sha(C['plan5'])==C['plan5_sha256'] and sha(C['plan10'])==C['plan10_sha256']
p5=read(C['plan5']);p10=read(C['plan10'])
collector_intent=visible(C['plan5']).parent/'collector-intent.json'
saved=read(collector_intent)
entries=[Path(arg) for arg in saved['argv'] if Path(arg).name in ('rollover_terminal.py','collector_entry.py')]
assert len(entries)==1 and saved['plan_sha256']==C['plan5_sha256']
helper=visible(entries[0].parent/'rollover_terminal.py')
assert sha(helper)==p5['source_code_sha256']
manifest=read(helper.parent/'manifest.json')
for name in ('rollover_terminal.py','rollover_paths.py','predecessor_generation_terminal.py'):
 assert sha(helper.parent/name)==manifest['files'][name]['sha256']
sys.path.insert(0,str(helper.parent));t=importlib.import_module('rollover_terminal')
t.validate_plan(p5);t.validate_plan(p10)
assert p5['node']=='h100-5' and p10['node']=='h100-10'
assert p5['remote_node10_plan_sha256']==C['plan10_sha256']
assert t.core.physical(p5['remote_node10_plan_path'])==t.core.physical(C['plan10'])
assert t.core.physical(p5['remote_node10_proof_path'])==t.core.physical(C['done10'])
done=t.verify_proof(C['done10'],C['done10_sha256'],'h100-10',C['plan10_sha256'])
assert t.core.physical(done['source_plan'])==t.core.physical(C['plan10'])
assert len(done['ordered_units'])==6
assert os.stat('/proc/self/ns/mnt').st_ino==p5['source_mount_ns_inode']
assert os.stat('/proc/self/ns/net').st_ino==p5['source_net_ns_inode']
rows=p5['ordered_units'];upstream='sst-q7b-retrieval-relay5-v1-'+t.RUN_ID[:12]+'.service'
assert len(rows)==9 and rows[-2]['unit']==upstream and rows[-1]['unit']==t.core.SOURCE['unit']
assert {r['unit'] for r in rows[:7]}=={t.unit(n) for n in t.SOURCE_MODELS[:-1]}
collector='sst-q7b-terminal-rollover-v1-h100-5-'+t.RUN_ID[:12]+'.service'
assert collector not in {r['unit'] for r in rows}
mygroup=Path('/proc/self/cgroup').read_text()
for row in rows:
 assert '/system.slice/'+row['unit'] not in mygroup
 t.verify_allocation(row)
def query(row):
 result=subprocess.check_output(['systemctl','show',row['unit'],'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup','-p','Result','-p','ExecMainStatus'],text=True,timeout=3)
 props=dict(line.split('=',1) for line in result.splitlines() if '=' in line)
 assert props.get('Id')==row['unit']
 return props
def terminal(row,props):
 if props.get('MainPID')!='0' or props.get('ActiveState') not in ('inactive','failed'):return False
 assert not t.core.owned('/system.slice/'+row['unit']),'Own cgroup is not empty'
 assert not {p:b for p,b in row['original_owned_births'].items() if t.core.birth(p)==b},'Original own birth remains live'
 return True
def active(row,props):
 assert props.get('MainPID')==str(row['pid']) and props.get('ActiveState')=='active','Refuse unknown/replaced/transitional unit'
 assert props.get('ControlGroup')=='/system.slice/'+row['unit']
 assert t.core.birth(row['pid'])==str(row['birth']),'Refuse changed original PID birth'
 return True
groups={'extras':rows[:7],'upstream':[rows[-2]],'root':[rows[-1]]}
prior={'extras':[],'upstream':rows[:7],'root':rows[:-1]}[C['step']]
root=visible(C['plan5']).parent/'operator-exact-retire-v1';root.mkdir(exist_ok=True)
binding=dict(run_id=t.RUN_ID,plan5=t.core.physical(C['plan5']),plan5_sha256=C['plan5_sha256'],
 plan10=t.core.physical(C['plan10']),plan10_sha256=C['plan10_sha256'],done10=t.core.physical(C['done10']),done10_sha256=C['done10_sha256'],
 operator_code_sha256=C['operator_code_sha256'],terminal_helper_sha256=sha(helper),collector_intent_sha256=sha(collector_intent))
with (root/'operator.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for row in prior:assert terminal(row,query(row)),'Prior exact own step is not truly terminal; do not retire its dependency'
 operation=root/(C['step']+'-intent.json')
 if operation.exists():
  existing=read(operation);assert existing['binding']==binding
  result=dict(status='EXISTING_ONCE_INTENT_OBSERVE_ONLY_NO_REPLAY',step=C['step'])
 else:
  before=[]
  for row in groups[C['step']]:
   props=query(row)
   if not terminal(row,props):active(row,props)
   before.append(dict(unit=row['unit'],pid=row['pid'],birth=row['birth'],observed=props))
  t.core.publish(operation,dict(binding=binding,step=C['step'],ordered_units=before,epoch=time.time(),
   preserves_all_partial_outputs=True,signals_collector=False,signals_foreign=False))
  for index,row in enumerate(groups[C['step']]):
   props=query(row);skip=terminal(row,props)
   if not skip:active(row,props)
   item=root/('%s-%02d-intent.json'%(C['step'],index))
   command=['systemctl','stop','--no-block',row['unit']]
   t.core.publish(item,dict(binding=binding,unit=row['unit'],original_pid=row['pid'],original_birth=row['birth'],observed_before=props,
    action='ALREADY_TERMINAL_NO_SIGNAL' if skip else 'EXACT_ORIGINAL_UNIT_STOP_NO_BLOCK',argv=None if skip else command,epoch=time.time()))
   outcome=dict(action='ALREADY_TERMINAL_NO_SIGNAL',returncode=None)
   if not skip:
    # Recheck original MainPID/birth immediately before the exact-unit request.
    now=query(row)
    if terminal(row,now):outcome['action']='BECAME_TERMINAL_NO_SIGNAL'
    else:
     active(row,now)
     try:
      r=subprocess.run(command,text=True,capture_output=True,timeout=5)
      outcome=dict(action='EXACT_ORIGINAL_UNIT_STOP_REQUESTED',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
     except subprocess.TimeoutExpired as error:
      outcome=dict(action='STOP_REQUEST_UNKNOWN_DO_NOT_REPLAY',error=str(error))
   outcome.update(unit=row['unit'],epoch=time.time(),observed_after=query(row))
   t.core.publish(item.with_name(item.name.replace('-intent.json','-receipt.json')),outcome)
   assert outcome['action']!='STOP_REQUEST_UNKNOWN_DO_NOT_REPLAY' and outcome.get('returncode') in (None,0),'Stop request failed/unknown; no dependent advance'
  result=dict(status='EXACT_OWN_STOP_REQUESTS_RECORDED',step=C['step'])
 observations=[]
 for row in groups[C['step']]:
  props=query(row);observations.append(dict(unit=row['unit'],observed=props,terminal=terminal(row,props)))
 result.update(binding=binding,observations=observations,all_step_units_terminal=all(x['terminal'] for x in observations),
  collector_not_signalled=True,original_partial_outputs_retained=True,terminal_proof_is_independent_collector=True,epoch=time.time())
 t.core.publish(root/('%s-observation-%d.json'%(C['step'],time.time_ns())),result)
 print(json.dumps(result))
'''

OUTER = r'''
import json, os, subprocess
from pathlib import Path
def birth(pid):
 try:return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except (FileNotFoundError,ProcessLookupError):return None
target=1490214
if birth(target)!='6807965':
 unit='sst-q7b-terminal-rollover-v1-h100-5-70188b8e50fd.service'
 text=subprocess.check_output(['systemctl','show',unit,'-p','Id','-p','MainPID','-p','ActiveState','-p','ControlGroup'],text=True,timeout=5)
 props=dict(line.split('=',1)for line in text.splitlines()if'='in line)
 assert props.get('Id')==unit and props.get('ActiveState')=='active' and props.get('ControlGroup')=='/system.slice/'+unit
 target=int(props['MainPID']);assert target>1 and birth(target) is not None
script='C='+repr(C)+'\n'+INNER
r=subprocess.run(['nsenter','--target',str(target),'--mount','--net','--','/usr/bin/python3','-'],input=script,text=True,capture_output=True,timeout=90)
print(r.stdout,end='');print(r.stderr,end='',file=__import__('sys').stderr)
raise SystemExit(r.returncode)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=('extras', 'upstream', 'root'), required=True)
    parser.add_argument('--plan5', default=BASE+'/preparation/cohort-rollover-v1-h100-5/plan.json')
    parser.add_argument('--plan10', default=BASE+'/preparation/cohort-rollover-v1-h100-10/plan.json')
    parser.add_argument('--plan5-sha256', required=True)
    parser.add_argument('--plan10-sha256', required=True)
    parser.add_argument('--done10', default=PHYSICAL+'/preparation/cohort-rollover-v1-h100-10/complete.json')
    parser.add_argument('--done10-sha256', default=DONE10_SHA)
    parser.add_argument('--execute', action='store_true', help='Actually send this one step; omitted means local preview only')
    args = parser.parse_args()
    for value in (args.plan5_sha256, args.plan10_sha256, args.done10_sha256):
        assert len(value)==64 and all(ch in '0123456789abcdef' for ch in value)
    config = vars(args).copy();config.pop('execute')
    config.update(base=BASE, physical=PHYSICAL, operator_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not args.execute:
        print(json.dumps(dict(status='LOCAL_PREVIEW_NO_REMOTE_ACTION',config=config,
            sequence=['extras: repeat same step observes only until all terminal', 'upstream: requires all seven extras terminal', 'root: requires all extras and upstream terminal'],
            terminal_collector='NEVER_STOPPED',partial_outputs='PRESERVED'),indent=2))
        return
    sys.path.insert(0,REPO)
    from reproduction.operations.qwen7b_resample import shared_vram_control as c
    script='C='+repr(config)+'\nINNER='+repr(INNER)+'\n'+OUTER
    local=Path(__file__).parent/('qwen7b-source5-exact-retire-actual-%d.json'%time.time_ns())
    try:
        result=c.transport(script,'h100-5')
        value=dict(config=config,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time())
    except Exception as error:
        value=dict(config=config,status='TRANSPORT_UNKNOWN_NO_AUTOMATIC_STOP_REPLAY',error=repr(error),epoch=time.time())
        with local.open('x') as stream:json.dump(value,stream,indent=2)
        print(str(local));raise
    with local.open('x') as stream:json.dump(value,stream,indent=2)
    print(str(local));print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
