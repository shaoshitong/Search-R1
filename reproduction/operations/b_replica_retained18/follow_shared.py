"""Read-only supervision through 200, retaining the five-outer milestone."""
import time,json
from .shared_observe import observe
from .control import OUT

def main():
 while True:
  v=observe()
  if v.get('complete_outer')==200:
   final=next((r for r in v['facts']if r['outer']==200),None)
   if not final or {r['rank']for r in final['ranks']}!=set(range(8))or not all(r['ledger_sha_matches']for r in final['ranks']):
    raise RuntimeError('Outer200 joint requires matching direct records from all eight ranks')
   # Observation is evidence for the main agent's final audit, not goal completion.
   (OUT/'SHARED_RESUME64_200_OBSERVED.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));return
  if v['cleanup']or not v.get('actual_node_health',{}).get('same_birth'):
   raise RuntimeError('Native shared writer exited; preserve evidence and diagnose')
  if (v.get('complete_outer')or 64)>=69 and(v['stage']or{}).get('outer',0)>=70:
   p={k:r for k,r in v['progress'].items()if 'outer-000070/'in k}
   milestone=OUT/'SHARED_RESUME64_FIVE_COMPLETE.json'
   if not milestone.exists()and sum(r.get('completed_trajectories',0)for r in p.values())>0:
    rows={r['outer']:r for r in v['facts']}
    if all(i in rows and {r['rank']for r in rows[i]['ranks']}==set(range(8))and all(r['ledger_sha_matches']and r['opportunities']==20 and r['writes']>=1 for r in rows[i]['ranks'])for i in range(65,70)):
     milestone.write_text(json.dumps(v,ensure_ascii=False,indent=2))
  time.sleep(35)

if __name__=='__main__':main()
