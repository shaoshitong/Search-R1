"""One bounded control action and small audit; no watcher or Slurm submission."""
import argparse,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--mode',choices=('status','verify','merge','audit'),default='status');a=p.parse_args()
here=pathlib.Path(__file__).resolve().parent
name={'status':'extra-g5-pilot-readonly-status.py','verify':'extra-g5-pilot-first-batch-verify.py',
      'merge':'extra-g5-pilot-merge-once.py','audit':'extra-g5-pilot-g10-independent-audit.py'}[a.mode]
r=subprocess.run(['python3',str(here/'balanced-control.py'),str(here/name),'--timeout','50'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=55)
values=[json.loads(line) for line in r.stdout.splitlines() if line.strip().startswith('{')]
if r.returncode or not values:raise RuntimeError('Fresh pilot control query failed, code='+str(r.returncode))
v=values[-1];v.setdefault('observed_epoch',v.get('epoch',time.time()));directory=here/'extra-g5-pilot-audits';directory.mkdir(exist_ok=True)
path=directory/('%s-%.6f.json'%(a.mode,v['observed_epoch']));path.write_text(json.dumps(v,indent=2)+'\n')
if a.mode!='status':out=v
else:
 out=dict(observed_epoch=v['observed_epoch'],queue=v['queue'],accounting=v['accounting'],shards={})
 for key,item in v['shards'].items():
  own={name:item[name] for name in ('progress.json','failure.json','complete.json','extra-g5-complete.json') if name in item}
  own['phases']=[dict(batch=batch['batch'],**{phase:batch[phase] for phase in ('prefix','full') if phase in batch}) for batch in item['batches']]
  out['shards'][key]=own
 out['logs']=v['logs']
out['saved_small_audit']=str(path)
print(json.dumps(out))
