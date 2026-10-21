"""External bounded pure-CPU paired stability after an accepted milestone audit.

Stage is code-only. Run never changes the trainer, model, labels or prediction
files; only one separately locked finite worker per already-audited milestone.
"""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,zlib
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
COMMIT='1a0f5b411e9c31ff9a3b64a02aa9644156eee6c0'
REMOTE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count-stability-1a0f5b4'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2'
p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','run','status'),required=True);p.add_argument('--cursor',type=int,choices=(256,512,768,938));a=p.parse_args()
if a.action!='stage' and a.cursor is None:p.error('Actual accepted milestone cursor required')
spec=importlib.util.spec_from_file_location('joint_count_stability_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
source='''import base64,fcntl,hashlib,json,os,pathlib,subprocess,sys,time,zlib
root=pathlib.Path(inputs['root']);assert root.resolve()==root and root.name=='direct-hidden-count-stability-1a0f5b4';root.mkdir(exist_ok=True)
def ref(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(pathlib.Path(p).read_text())
def immutable(p,v):
 if p.exists():assert read(p)==v,'Existing immutable output differs'
 else:
  tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,sort_keys=True,indent=2)+'\\n');tmp.replace(p)
code=root/'paired_question_stability.py'
if inputs['action']=='stage':
 payload=zlib.decompress(base64.b64decode(inputs['payload']));assert len(payload)==inputs['bytes'] and hashlib.sha256(payload).hexdigest()==inputs['sha256']
 if code.exists():assert code.read_bytes()==payload
 else:code.write_bytes(payload);code.chmod(0o444)
 immutable(root/'source-publication.json',dict(commit=inputs['commit'],source=ref(code),no_GPU_or_data_operation=True));print(json.dumps(dict(status='IMMUTABLE_PAIRED_STABILITY_CPU_SOURCE_STAGED',source=ref(code),commit=inputs['commit'])));sys.exit(0)
pub=read(root/'source-publication.json');assert pub['commit']=='1a0f5b411e9c31ff9a3b64a02aa9644156eee6c0' and pub['source']==ref(code)
cursor=inputs['cursor'];base=pathlib.Path(inputs['base']);assert base.name=='direct-hidden-count30016-train-v2';milestone=base/'evaluations'/('cursor-%06d'%cursor);work=root/('cursor-%06d-v2'%cursor);work.mkdir(exist_ok=True)
if inputs['action']=='status':
 result=dict(status='PAIRED_CPU_CURRENT_STATE',cursor=cursor,time=time.time())
 for name in ('controller.json','worker-exit.json'):
  if (work/name).exists():result[name[:-5]]=read(work/name)
 if (work/'result.json').exists():
  v=read(work/'result.json');result['result_ref']=ref(work/'result.json');result['status']=v['status'];result['scopes']={k:{'models':x['models'],'comparisons':x['comparisons'],'bootstrap_replicates_used':x['bootstrap_replicates_used']} for k,x in v['scopes'].items()};result['strata_SHA']=v['strata']['strata_sha256']
 if (work/'worker.log').exists():result['log_tail']=(work/'worker.log').read_text()[-1500:]
 print(json.dumps(result));sys.exit(0)
lock=(work/'operator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (work/'controller.json').exists():print(json.dumps(dict(status='EXISTING_FINITE_STABILITY_WORKER_NO_DUPLICATE',controller=read(work/'controller.json'))));sys.exit(0)
audit_path=milestone/'independent-audit.json';audit=read(audit_path);assert audit['status']=='JOINT_HIDDEN_COUNT_FIXED_IDS_OWN_EM_DIRECT_Q_METRICS_FULL_CP_INDEPENDENTLY_ACCEPTED' and audit['cursor']==cursor and audit['threshold']==.5 and audit['threshold_fitted'] is False
assert audit['all_nine_states_unchanged_during_evaluation'] is True
c=read(base/'config.json');backend=c['validation_reader_audit'];assert ref(backend['path'])['sha256']==backend['sha256']
candidate=audit['predictions']['selection2048FreshG5HF']
E2=dict(path=str(base.parent/'E2-same-HF-baseline-20261021/predictions.jsonl.gz'),bytes=155453,sha256='e0dc53be8c0847e8a736401452d309771bcdca98d5c129e6cef49afa3b12c2ac')
E4=dict(path=str(base.parent/'peer-newseed8192-g10-train-v1/E4-same-HF-validation/E4/predictions.jsonl.gz'),bytes=155299,sha256='66cdf3e7c748bda7f508c4f381ad0456a413d5d73c73e5d9f1eb774b71bdb707')
for value in (candidate,E2,E4):assert ref(value['path'])==value
manifest=dict(role='already_viewed_recipe_selection_validation',questions=2048,bootstrap_replicates=2000,predictions=dict(candidate=candidate,E2=E2,E4=E4),independent_identity_audit=ref(audit_path),same_HF_backend_receipt=ref(backend['path']),threshold=.5,seed=9137,fitting_allowed=False,external_final_read=False)
immutable(work/'manifest.json',manifest)
# The pure-stdlib worker sets affinity2, address-space3GB, CPU180s itself.
args=['python3',str(code),'--manifest',str(work/'manifest.json'),'--output',str(work/'result.json')]
worker="import subprocess,json,pathlib,time; a="+repr(args)+"; p=subprocess.run(a); pathlib.Path("+repr(str(work/'worker-exit.json'))+").write_text(json.dumps(dict(returncode=p.returncode,time=time.time()))); raise SystemExit(p.returncode)"
compile(worker,'paired-worker','exec')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(work/'worker.log').open('ab',0)
child=subprocess.Popen(['python3','-c',worker],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
immutable(work/'controller.json',dict(pid=child.pid,source=pub,manifest=ref(work/'manifest.json'),CPU_only=True,bounded_seconds=180,time=time.time()));print(json.dumps(dict(status='FINITE_PAIRED_CPU_STABILITY_STARTED',cursor=cursor,pid=child.pid,manifest=ref(work/'manifest.json'))))
'''
inputs=dict(root=REMOTE,base=BASE,action=a.action,cursor=a.cursor)
if a.action=='stage':
 name='reproduction/operations/qwen7b_warmup/paired_question_stability.py';payload=subprocess.check_output(['git','show',COMMIT+':'+name],cwd=ROOT)
 inputs.update(commit=COMMIT,payload=base64.b64encode(zlib.compress(payload,9)).decode(),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
command='python3 -c '+shlex.quote('inputs='+repr(inputs)+'\n'+source)
if len(command.encode())>=16000:raise ValueError('Bounded control command exceeds16KB')
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=45)
if r.returncode:raise RuntimeError(r.stderr[-2500:])
v=json.loads(r.stdout);(OUT/('actual-paired-'+a.action+'-'+str(a.cursor)+'.json')).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
