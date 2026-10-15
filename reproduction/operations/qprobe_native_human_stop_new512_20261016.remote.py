"""Executed remote exact-birth human stop; stdin supplies bound human intent.

Archived operational source, not an automatic retry or launcher.
"""

import os,sys,json,hashlib,signal,time,socket,subprocess
from pathlib import Path
p=json.load(sys.stdin);base=Path(p['intent']['base']);repo=base/'repo'
assert base==base.resolve() and base.stat().st_uid==os.getuid()
def read(n):
 q=base/n;assert q==q.resolve() and q.stat().st_uid==os.getuid() and q.stat().st_size<256*1024**2
 return json.loads(q.read_text())
run=read('logs/run-manifest.json');claim=read('logs/pipeline-launch-claim.json');launch=read('logs/deployment-launch-receipt.json')
for key in ('run_id','execution_id','source_commit'):assert run[key]==p['human']['target_'+key]==launch[key]
assert claim['run_id']==run['run_id'] and claim['execution_id']==run['execution_id']
assert run['source_sha256']==p['intent']['source_sha256'] and run['execution_host']==socket.gethostname()
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==run['source_commit']
relative='reproduction/a800/owned_processes.py';assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==run['source_sha256'][relative]
sys.path.insert(0,str(repo));from reproduction.a800.owned_processes import same,identity,signal_owned
producer=run['pipeline_identity'];assert producer==claim['pipeline_identity']
assert (producer['pid'],producer['start_ticks'],producer['sid'],producer['uid'])==(340952,616000142,340952,os.getuid())
assert (launch['pid'],launch['start_ticks'],launch['sid'])==(340952,616000142,340952)
assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
ledger=read('logs/process-ledger.json');assert ledger['run_id']==run['run_id']
record=dict(role='human_native_probe_diagnostic_stop_intent',run_id=run['run_id'],execution_id=run['execution_id'],source_commit=run['source_commit'],human_instruction_sha256=p['human_sha256'],human_recorded_unix=p['human']['recorded_unix'],pipeline_identity=producer,owned_job_names=list(ledger['jobs']),automatic_recovery_authorized=False,checkpoints_preserved=True,checked_unix=time.time())
def once(n,data):
 q=base/n
 with q.open('x') as f:json.dump(data,f,sort_keys=True);f.flush();os.fsync(f.fileno())
previous=read('preparation/human-native-new512-stop-intent.json');assert previous['human_instruction_sha256']==p['human_sha256'] and previous['pipeline_identity']==producer;assert not (base/'logs/human-native-new512-stop-signal.json').exists()
sent=False
if same(producer):
 a=identity(producer['pid']);argv=(Path('/proc')/str(producer['pid'])/'cmdline').read_bytes().split(b'\0');assert [x.decode() for x in argv if x]==['/data/search-g1/env/bin/python','-m','reproduction.a800_common.qprobe_qwen_formal_ops','--base',str(base),'--action','pipeline']
 if a['state'] not in ('Z','X'):signal_owned(producer,signal.SIGTERM);sent=True
result=dict(record,signal_sent=sent,signal='SIGTERM' if sent else None,cleaned_not_yet_verified=True)
once('logs/human-native-new512-stop-signal.json',result)
print(json.dumps(result),flush=True)
