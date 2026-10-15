
import json,sys,pathlib,time,os,signal
p=json.load(sys.stdin);b=pathlib.Path(p['base']);sys.path.insert(0,p['runtime_repo'])
from reproduction.a800.owned_processes import identity,same
launch=json.loads((b/'logs/launch.json').read_text());q=launch['process'];s=json.loads((b/'logs/status.json').read_text())
assert q==dict(pid=425107,start_ticks=616579516,sid=425107,uid=os.getuid())
assert same(q) and identity(q['pid'])['state'] not in ('Z','X')
assert s['action']=='features' and s['actor_adam_updates']==0 and not (b/'warmup').exists(), 'Fit already advanced: inspect instead of repeating'
claim=b/'logs/human-balanced-sampling-transition.json'
with claim.open('x') as f:json.dump(dict(raw_latest_human='可以动态采样，保证比例相等这样子。',scope='hold only supervisor before old-unbalanced fit; existing prefix encoder continues; new balanced fit required',process=q,feature_child=s['child'],claimed_unix=time.time(),actor_training_remains_stopped=True),f)
os.kill(q['pid'],signal.SIGSTOP)
time.sleep(.05)
actual=identity(q['pid']);assert actual['state']=='T' and same(q)
v=dict(role='latest_balanced_sampling_supervisor_hold',process=q,supervisor_state=actual['state'],feature_child=s['child'],feature_child_alive=same(s['child']),signal='SIGSTOP',encoder_not_stopped=True,fit_not_started=True,checked_unix=time.time())
with (b/'logs/human-balanced-sampling-hold-actual.json').open('x') as f:json.dump(v,f)
print(json.dumps(v),flush=True)

