"""Cancel only two human-authorized, independently confirmed idle Easy services.
No retriever query, GPU forward, training restart or asset deletion is performed.
"""
import json,subprocess,time,urllib.request
from pathlib import Path
specs=[dict(job='745198',run='976e20f47ae14f9aa9c65010a0ec0b97',url='http://ACD1-26:18020/health',base='/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021',completed_queries=1094682),dict(job='750471',run='a8ddc12aec1840d898e4d9cfcabc17c1',url='http://ACD1-4:18026/health',base='/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-batch-production-r1-20261023',completed_queries=2405909)]
def cmd(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=20);return dict(args=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));receipts=[]
# Read both identities and idle counters before any cancellation.
for s in specs:
 q=cmd(['squeue','-h','-j',s['job'],'-o','%i|%u|%j|%T|%N']);assert q['returncode']==0,q
 r=dict(job=s['job'],before_queue=q,authorization='explicit user coordination request; both threads confirm no remaining consumers',must_keep_jobs=['751011','761991'],checked_unix=time.time(),files_deleted=False)
 if q['stdout'].strip():
  fields=q['stdout'].strip().split('|');assert fields[0]==s['job'] and fields[1]=='jxu957',fields
  with opener.open(s['url'],timeout=10) as response:h=json.load(response)
  assert str(h['service_job_id'])==s['job'] and h['service_run_id']==s['run'],h.get('service_job_id')
  stats=h['statistics'];assert stats['completed_queries']==s['completed_queries'],'new consumer queries observed; do not cancel'
  assert stats['logical_queries']==stats['completed_queries'],'in-flight query observed; do not cancel'
  r['health_before']={k:h.get(k) for k in ['service_job_id','service_run_id','source_commit','endpoint','physical_gpu_count','statistics']}
 receipts.append(r)
for s,r in zip(specs,receipts):
 evidence=Path(s['base'])/'preparation/human-idle-stop-20261025';evidence.mkdir(exist_ok=True);p=evidence/'receipt.json';p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 for log in (Path(s['base'])/'logs').glob('service-'+s['job']+'.log'):
  with log.open('rb') as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-65536));tail=f.read()
  (evidence/(log.name+'.tail')).write_bytes(tail)
 if r['before_queue']['stdout'].strip():
  r['scancel']=cmd(['scancel',s['job']]);assert r['scancel']['returncode']==0,r['scancel']
 else:r['scancel']=dict(skipped=True,reason='already not queued')
 r['after_queue']=cmd(['squeue','-h','-j',s['job'],'-o','%i|%u|%T|%N']);r['accounting']=cmd(['sacct','-j',s['job'],'-nP','-o','JobIDRaw,User,State,ExitCode,Elapsed,NodeList']);r['checked_unix']=time.time();p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(checked_unix=time.time(),services=receipts,cancelled_job_ids=[s['job'] for s in specs],must_keep_untouched=['751011','761991'],files_deleted=False,retrieval_queries_sent=0,gpu_forward_triggered=False)))
