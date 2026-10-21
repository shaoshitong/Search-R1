import json,pathlib,subprocess,sys,time,concurrent.futures
C=pathlib.Path(__file__).resolve().parents[3];R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019/reports'
def sample(n):
 pub=json.loads((O/('joint-audited'+n+'-publication.json')).read_text());cmd=['python3','-m','reproduction.operations.qwen7b_warmup.deploy_peer_joint_subset_l26','--control-root',str(C),'--commit',pub['commit'],'--branch',pub['branch'],'--config',pub['config'],'--action','status'];p=subprocess.run(cmd,cwd=R,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=75)
 if p.returncode:raise RuntimeError(p.stderr[-1500:]+p.stdout[-1500:])
 v=json.loads([x for x in p.stdout.splitlines()if x.startswith('{')][-1]);out=dict(subset=n,epoch=v['epoch'],squeue=v.get('squeue'),sacct=v.get('sacct'),shards={i:{k:x[k]for k in ('complete_chunks','progress.json','failure.json','device.json','complete')if k in x}for i,x in v['shards'].items()})
 if 'failure.json'in v:out['failure']=v['failure.json']
 if 'preparation'in v:out['preparation']=v['preparation']
 elif 'prepare-progress.json'in v:out['prepare_progress']=v['prepare-progress.json']
 for pair,own in v['pairs'].items():
  for name,text in own['logs'].items():
   if any(e in text for e in ('Traceback (','RuntimeError:','ValueError:','CUDA out of memory','CANCELLED','error:')):out.setdefault('log_errors',{})[name]=text[-1600:]
 with(O/('joint-audited'+n+'-actual-feature-observations.jsonl')).open('a')as f:f.write(json.dumps(out,sort_keys=True)+'\n')
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
 for out in pool.map(sample,sys.argv[1:]):print(json.dumps(out),flush=True)
