"""Bounded code-only receiver staging after real SSH recovery; never receive."""
import ast,concurrent.futures,hashlib,importlib.util,json,os,shlex,stat,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
MODULE=REPO/'reproduction/operations/qwen7b_warmup/peer_newseed_receive.py'
PUB='755ed4ba3eef237ab2244c09c17cdac3966a43fc';SNAPSHOT='newseed-20261021-185747-445fc691'
assert MODULE.read_bytes()==subprocess.check_output(['git','show',PUB+':'+str(MODULE.relative_to(REPO))],cwd=str(REPO))
spec=importlib.util.spec_from_file_location('receiver',str(MODULE));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
preflight=None
for node in ast.parse((HERE/'qualify-peer-newseed-receiver.py').read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='preflight' for x in node.targets):preflight=ast.literal_eval(node.value)
assert isinstance(preflight,str)

def save(name,value):
 p=HERE/name;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');os.replace(str(t),str(p))

def one(item):
 label,argv=item;t=time.time()
 try:
  r=subprocess.run(argv+['hostname'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=12)
  return dict(route=label,returncode=r.returncode,hostname=r.stdout.decode().strip() if not r.returncode else None,seconds=time.time()-t),argv
 except Exception as error:return dict(route=label,error_type=type(error).__name__,seconds=time.time()-t),argv

for attempt in range(1,7):
 started=time.time();ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 common=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=6']
 routes=[('alias-hkust2',common+['hkust2']),('en0-hkust2',common+['-B','en0','-b',ip,'-oProxyCommand=none','-oProxyJump=none','hkust2']),('alias-hkust3',common+['hkust3']),('en0-hkust3',common+['-B','en0','-b',ip,'-oProxyCommand=none','-oProxyJump=none','hkust3'])]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:observed=list(pool.map(one,routes))
 record=dict(attempt=attempt,epoch=time.time(),routes=[x[0] for x in observed],receiver_started=False,Mac_payload_transit=False)
 with (HERE/'peer-newseed-receiver-reachability-observations.jsonl').open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')
 save('peer-newseed-receiver-reachability-latest.json',record);print(json.dumps(record),flush=True)
 ready=[(x,argv) for x,argv in observed if x['route'].endswith('hkust2') and x.get('returncode')==0 and x['hostname']=='xielab2']
 if ready:
  ssh=ready[0][1];code=m.install_code(ssh,m.K2/SNAPSHOT)
  probe='import pathlib,json,hashlib,time; p=pathlib.Path(%r); b=p.parents[2]; print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),receiver_started=(b/"receiver.lock").exists(),catalog_exists=(b/"catalog.json").exists(),complete_exists=(b/"complete.json").exists(),epoch=time.time())))'%str(code)
  first=m.remote(ssh,'python3 -c '+shlex.quote(probe),timeout=30)
  assert first['sha256']==hashlib.sha256(MODULE.read_bytes()).hexdigest() and not first['receiver_started'] and not first['catalog_exists'] and not first['complete_exists']
  save('peer-newseed-receiver-HKUST2-stage.json',dict(source_commit=PUB,route=ready[0][0]['route'],**first))
  token=Path.home()/'.cache/huggingface/token';assert stat.S_IMODE(token.stat().st_mode)==0o600
  metadata=m.remote(ssh,'python3 -c '+shlex.quote(preflight),dict(hf_token=token.read_text().strip()),timeout=55)
  save('peer-newseed-receiver-HKUST2-HF-preflight.json',metadata)
  second=m.remote(ssh,'python3 -c '+shlex.quote(probe),timeout=30)
  assert second['sha256']==first['sha256'] and not second['receiver_started'] and not second['catalog_exists'] and not second['complete_exists']
  save('peer-newseed-receiver-HKUST2-stage-observation2.json',second)
  result=dict(status='PUBLISHED_RECEIVER_CODE_STAGED_ONLY_AND_METADATA_PREFLIGHT_COMPLETE',source_commit=PUB,code=first,metadata=metadata,second_fresh_code_verification=True,receiver_started=False,epoch=time.time())
  save('peer-newseed-receiver-preparation-complete.json',result);print(json.dumps(result),flush=True);break
 if attempt==6:
  save('peer-newseed-receiver-preparation-incomplete.json',dict(status='HKUST2_SSH_UNAVAILABLE_NO_RECEIVER_STARTED',attempts=attempt,last=record,epoch=time.time()));break
 # The separate exec session remains interruptible; model is never held in a
 # synchronous >60sec wait and root can inspect the bounded small observations.
 time.sleep(max(1,60-(time.time()-started)))
