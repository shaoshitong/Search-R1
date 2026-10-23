"""Read-only CPU acceptance of a real periodic probe and stale-wrapper resolution.

This uses the active runtime's recovery function with real saved metadata; it
neither injects a production failure nor claims a GPU reload has occurred.
"""
import argparse
from reproduction.dynrpo7b.local_control import remote

AUDIT = r'''
import hashlib,json,pathlib,sys,torch
from reproduction.dynrpo7b.runtime import restored_probe_metadata
from reproduction.dynrpo7b.prefix_schedule import DEFAULT_SCHEDULE
b=pathlib.Path(BASE);d=b/'artifacts/question-probe'/('update-%04d'%OUTER)
p=d/'published.json'
if not p.exists():
 print(json.dumps(dict(status='WAITING_FOR_PUBLISHED_PROBE',outer=OUTER)))
 sys.exit(0)
publication=json.loads(p.read_text());ref=publication['checkpoint']
expected=DEFAULT_SCHEDULE.update_binding(OUTER)
assert publication['status']=='PROBE_UPDATE_PUBLISHED'
assert all(publication[k]==v for k,v in expected.items())
epochs=[json.loads((d/('epoch-%02d.json'%n)).read_text()) for n in (1,2,3)]
assert [r['epoch'] for r in epochs]==[1,2,3]
assert all(r['actual_questions']==2560 and r['actual_optimizer_updates']==20 for r in epochs)
assert all(r['dataset_sha256']==publication['dataset_sha256'] for r in epochs)
assert epochs[2]['online_updates']-epochs[0]['online_updates']==40
h=hashlib.sha256()
with pathlib.Path(ref['path']).open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==ref['sha256']
state=torch.load(ref['path'],map_location='cpu',weights_only=False,mmap=True)
assert state['format']=='DYNRPO7B_GROUP_VARIANCE_PROBE_V1'
assert state['completed_update_outer']==OUTER and state['pending_update'] is None
assert state['online_updates']==publication['online_updates']==epochs[-1]['online_updates']
assert state['actual_Adam_updates']==state['warmup_cursor']+state['online_updates']
assert len(state['RNG_by_rank'])==4 and all(state['RNG_by_rank'])
assert state['raw'].keys()==state['EMA'].keys() and state['Adam']['state']
assert all(state['raw'][key].shape==state['EMA'][key].shape for key in state['raw'])
wrapper=b/'checkpoints/search-g1-common8/actor'/('global_step_%d'%OUTER)/'training_state/question-probe-state.pt'
old=torch.load(wrapper,map_location='cpu',weights_only=False)
assert old['completed_outer']==OUTER
assert state['warmup_cursor']==old['reference']['warmup_updates']
assert state['online_updates']==old['reference']['online_updates']+60
assert len(old['fit_groups'])==len({str(g['qid']) for g in old['fit_groups']})==2560
phase=dict(completed_actor_outer=OUTER,probe_completed_epoch=3,probe_pending=False,probe_ref=ref)
resolved=restored_probe_metadata(old,phase)
assert resolved['reference']==ref and resolved['probe_pending'] is False
assert resolved['head_version']=='seed1-E9-raw-online-%04d'%OUTER
assert old['reference']['sha256']!=ref['sha256']
steps=sorted(set(int(v['step']) for v in state['Adam']['state'].values() if 'step' in v))
assert steps==[state['actual_Adam_updates']]
r=dict(status='REAL_PROBE_FILE_AND_STALE_WRAPPER_RESOLUTION_PASS',outer=OUTER,
 checkpoint_sha256=ref['sha256'],epochs=[dict(epoch=x['epoch'],loss=x['target4variance_MSE'],
 questions=x['actual_questions'],online_updates=x['online_updates'],seconds=x['seconds']) for x in epochs],
 actual_Adam_updates=state['actual_Adam_updates'],warmup_updates=state['warmup_cursor'],
 online_updates=state['online_updates'],previous_online_updates=old['reference']['online_updates'],
 inherited_exactly_60_new_Adam_updates=True,EMA_updates=state['EMA_updates'],Adam_parameter_step_values=steps,
 RNG_ranks=len(state['RNG_by_rank']),raw_and_EMA_parameters=len(state['raw']),
 old_wrapper_sha=old['reference']['sha256'],resolved_new_sha=resolved['reference']['sha256'],
 stale_wrapper_test='CPU metadata resolution using actual saved pre-fit wrapper and epoch3 reference',
 GPU_reload_performed=False,production_failure_injected=False)
boundary=b/'preparation/dynrpo7b-checkpoints'/('global_step_%d'%OUTER)/'boundary.json'
if boundary.exists():
 row=json.loads(boundary.read_text())
 assert row['question_probe']['reference']['sha256']==ref['sha256']
 assert row['phase']['probe_pending'] is False and row['phase']['probe_completed_epoch']==3
 r['boundary_publication']='SAME_EPOCH3_REFERENCE'
print(json.dumps(r))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer',type=int,default=10)
    parser.add_argument('--base',default='/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r6')
    args=parser.parse_args()
    code='BASE='+repr(args.base)+'\nOUTER='+repr(args.outer)+'\n'+AUDIT
    remote('import subprocess,os\np=subprocess.run('+repr([args.base+'/env/bin/python','-c',code])+',cwd='+repr(args.base+'/repo')+',stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=90)\nprint(p.stdout);print(p.stderr[-2000:]);raise SystemExit(p.returncode)')

if __name__=='__main__':main()
