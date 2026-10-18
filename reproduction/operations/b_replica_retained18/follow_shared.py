"""Foreground verification of restored65..69 and actual positive outer70."""
import time,json
from .shared_observe import observe
from .control import OUT

def main():
 while True:
  v=observe()
  if v['cleanup']or not v.get('actual_node_health',{}).get('same_birth'):
   raise RuntimeError('Native shared writer exited; preserve evidence and diagnose')
  if (v.get('complete_outer')or 64)>=69 and(v['stage']or{}).get('outer',0)>=70:
   p={k:r for k,r in v['progress'].items()if 'outer-000070/'in k}
   if sum(r.get('completed_trajectories',0)for r in p.values())>0:
    (OUT/'SHARED_RESUME64_FIVE_COMPLETE.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));return
  time.sleep(35)

if __name__=='__main__':main()
