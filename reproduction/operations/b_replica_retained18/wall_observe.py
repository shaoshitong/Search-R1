"""Read native iteration totals; never substitute training time for a total."""
import json,time
from pathlib import Path
from .control import OUT,remote
from .shared_observe import BASE

def main():
 script='BASE='+repr(BASE)+'\n'+'''
import json,time
from pathlib import Path
p=Path(BASE);rows=[]
for outer in range(65,201):
 f=p/'outers'/('outer-%06d'%outer)/'iteration-wall-time.json'
 if f.exists():rows.append(json.loads(f.read_text()))
print(json.dumps(dict(observed_unix=time.time(),base=BASE,rows=rows)))
'''
 v=json.loads(remote(script,OUT/('iteration-wall-'+str(time.time_ns())+'-transport.json'),target='h100-10',established_controller=True))
 root=OUT.parents[1]
 for dest in (OUT,root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018')):
  (dest/'ITERATION_WALL_TOTALS.json').write_text(json.dumps(v,ensure_ascii=False,indent=2))
 print(json.dumps(dict(observed_unix=v['observed_unix'],count=len(v['rows']),latest=v['rows'][-1]if v['rows']else None)))

if __name__=='__main__':main()
