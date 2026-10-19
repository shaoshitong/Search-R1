"""Read-only supervision through 200, retaining the five-outer milestone."""
import time,json,re
from .shared_observe import observe
from .control import OUT

def main():
 while True:
  try:
   v=observe()
  except RuntimeError as error:
   # Transport failure does not establish that the native writer exited.
   # Keep scientific/ledger failures fatal instead of treating them as SSH loss.
   message=str(error)
   if not re.search(r'BOUNDED_ESTABLISHED_ROUTE_TIMEOUT|Timeout, server [^\r\n]+ not responding\.|Connection to [^\n]+ port \d+ timed out|ssh: connect to host [^\n]+|kex_exchange_identification:|Connection reset by peer|Connection closed by [^\n]+|Broken pipe',message,re.I):
    raise
   failure=dict(observed_unix=time.time(),status='CURRENT_RUNTIME_UNKNOWN_TRANSPORT_FAILURE',
    error=message,retry_after_seconds=1800,native_action='none')
   try:
    last=json.loads((OUT/'SHARED_RESUME64_LATEST.json').read_text())
    failure['last_confirmed']={k:last.get(k)for k in('observed_unix','complete_outer','stage','evidence')}
   except(OSError,ValueError):
    failure['last_confirmed']=None
   path=OUT/('shared-monitor-transport-failure-'+str(time.time_ns())+'.json')
   path.write_text(json.dumps(failure,ensure_ascii=False,indent=2))
   print(json.dumps(failure,ensure_ascii=False),flush=True)
   time.sleep(1800)
   continue
  if v.get('complete_outer')==200:
   final=next((r for r in v['facts']if r['outer']==200),None)
   if not final or {r['rank']for r in final['ranks']}!=set(range(8))or not all(r['ledger_sha_matches']for r in final['ranks']):
    raise RuntimeError('Outer200 joint requires matching direct records from all eight ranks')
   # Observation is evidence for the main agent's final audit, not goal completion.
   (OUT/'SHARED_RESUME64_200_OBSERVED.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));return
  if v['cleanup']or not v.get('actual_node_health',{}).get('same_birth'):
   raise RuntimeError('Native shared writer exited; preserve evidence and diagnose')
  start=v['resume_outer'];after_five=start+5;next_outer=after_five+1
  if (v.get('complete_outer')or start)>=after_five and(v['stage']or{}).get('outer',0)>=next_outer:
   p={k:r for k,r in v['progress'].items()if ('outer-%06d/'%next_outer)in k}
   milestone=OUT/('SHARED_RESUME%d_FIVE_COMPLETE.json'%start)
   if not milestone.exists()and sum(r.get('completed_trajectories',0)for r in p.values())>0:
    rows={r['outer']:r for r in v['facts']}
    if all(i in rows and {r['rank']for r in rows[i]['ranks']}==set(range(8))and all(r['ledger_sha_matches']and r['opportunities']==20 and r['writes']>=1 for r in rows[i]['ranks'])for i in range(start+1,after_five+1)):
     milestone.write_text(json.dumps(v,ensure_ascii=False,indent=2))
  time.sleep(35)

if __name__=='__main__':main()
