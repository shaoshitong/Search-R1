import json,os,re,subprocess,time
from pathlib import Path
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/native4-clip-native4-paired-r1-20261023')
job='749812';commit='2d5183455551145c52d1783a38663e2fcf170ceb'
assert os.getuid()==14090 and (root/'job.id').read_text().strip()==job
submitted=json.loads((root/'submit-claim.json').read_text());assert submitted['source_commit']==commit and submitted['mode']=='clip-native4'
result_path=root/'allow-historical-node-result.json';claim_path=root/'allow-historical-node-claim.json'
allowed={'JobId','UserId','JobState','Reason','ExcNodeList','NodeList','SchedNodeList','StartTime','ReqTRES','AllocTRES','Partition','TimeLimit'}
def query():
 r=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=20)
 assert r.returncode==0,r.stderr
 return {k:v for k,v in re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)',r.stdout) if k in allowed}
def publish(path,row):
 with path.open('x') as stream:json.dump(row,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
if result_path.exists():
 print(json.dumps({'status':'EXISTING_RESULT_INSPECT_NOT_REPLAY','result':json.loads(result_path.read_text()),'current':query()}),flush=True)
elif claim_path.exists():
 print(json.dumps({'status':'PRIOR_UPDATE_CLAIM_INSPECT_NOT_REPLAY','current':query()}),flush=True)
else:
 before=query();assert before['JobId']==job and before['UserId']=='jxu957(14090)'
 if before['JobState']!='PENDING' or before.get('ExcNodeList')!='ACD1-13':
  print(json.dumps({'status':'STATE_CHANGED_NO_MUTATION','before':before}),flush=True)
 else:
  command=['scontrol','update','JobId='+job,'ExcNodeList=']
  claim={'epoch':time.time(),'job':job,'source_commit':commit,'before':before,'command':command,'production_modified':False,'new_job_submitted':False,'reason':'Allow independent existing diagnostic to test current four-GPU allocation on formerly affected node; historical749590 had four allocated but three visible GPUs. Scheduler now has four free there; not claiming CUDA health or test acceptance. No healthy production stop or hot edit.'}
  publish(claim_path,claim)
  r=subprocess.run(command,capture_output=True,text=True,timeout=20)
  result={**claim,'status':'EXCLUSION_CLEAR_REQUEST_RETURNED','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'after':query(),'returned_epoch':time.time()}
  publish(result_path,result);print(json.dumps(result),flush=True)
  if r.returncode:raise SystemExit(r.returncode)
