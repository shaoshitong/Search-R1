"""Launch only an actually committed milestone5's CPU readback; no training mutation."""
import json,subprocess,importlib.util,datetime
from pathlib import Path
spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
arms=json.loads(Path('work/prefix5-ab-assigned-arms.json').read_text())
commit='2e3976230ac9425d66d8553afa9962821e964932'
wrapper='/mnt/shared/workspace/training_data_node_8/shaoshitong/search-g1-prefix5-reference-preparation-20261018/code/'+commit+'/repo/reproduction/operations/h100_reference_replica/h10011_private_paths.sh'
inner=r'''import json,sys,os,subprocess,time
from pathlib import Path
ARMS=__ARMS__
r={}
for s,a in ARMS.items():
 b=Path(a['base']);prep=b/'preparation';result=prep/'ab-joint-0005-independent-full-sha-cpu.json';receipt=prep/'ab-cp5-cpu-control-receipt.json';intent=prep/'ab-cp5-cpu-control-intent.json'
 if result.exists():r[s]={'status':'EXISTING_PROOF_NO_REPLAY','proof':str(result)};continue
 if receipt.exists():
  v=json.loads(receipt.read_text());r[s]=dict(v,pid_exists=Path('/proc/'+str(v['pid'])).exists(),proof_exists=False);continue
 if intent.exists():r[s]={'status':'EXISTING_INTENT_UNKNOWN_QUERY_NO_REPLAY'};continue
 if not(b/'outers/outer-000005/complete.json').exists():r[s]={'status':'PENDING_NOT_COMMITTED5_NO_AUDIT'};continue
 with intent.open('x')as f:json.dump({'source_commit':'2e3976230ac9425d66d8553afa9962821e964932','outer':5,'created_unix':time.time(),'cpu_only':True,'training_not_stopped':True},f)
 env=dict(os.environ,PYTHONPATH=str(b/'repo'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 command=['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-m','reproduction.prefix_selection200.ab_checkpoint_once','--base',str(b),'--outer','5']
 with(prep/'ab-cp5-cpu-control.log').open('ab')as log:p=subprocess.Popen(command,env=env,cwd=b/'repo',stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 st=Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()
 v={'status':'CPU_SAVEPOINT5_READBACK_STARTED_NOT_PASS','pid':p.pid,'start_ticks':int(st[19]),'outer':5,'log':str(prep/'ab-cp5-cpu-control.log'),'proof':str(result),'started_unix':time.time()}
 with receipt.open('x')as f:json.dump(v,f)
 r[s]=v
print(json.dumps(r))
'''.replace('__ARMS__',repr(arms))
script='import subprocess,os\np=subprocess.run(["unshare","--mount","--propagation","private","bash",'+repr(wrapper)+',"/usr/bin/python3","-c",'+repr(inner)+'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),text=True,capture_output=True,timeout=40)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
p=subprocess.run(ctl.route('h100-11','/usr/bin/python3 -'),input=script,text=True,capture_output=True,timeout=70);tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');v={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr};Path('work/prefix5-ab-cp5-cpu-control-'+tag+'.json').write_text(json.dumps(v,indent=2));assert p.returncode==0,p.stderr;print(p.stdout)
