"""Code-only qualification now; guarded exact continuation after real USR1."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
from . import direct_hidden_resume as r
BRANCH='codex/direct-count-exact-resume-20261021'
OWN=('direct_hidden_resume.py','test_direct_hidden_resume.py','deploy_direct_hidden_resume.py')
COMMON=r'''
import pathlib,json,hashlib,base64,zlib,subprocess,os,time,fcntl
b=pathlib.Path(inputs['base']);attempt=b/'continuations/attempt-0001-v2';attempt.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ref(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def atomic(p,x):
 p=pathlib.Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,sort_keys=True));tmp.replace(p)
def command(args):return subprocess.check_output(args).decode()
def publication():
 p=read(attempt/'operator-publication.json');source=pathlib.Path(p['code_root'])/'direct_hidden_resume.py'
 for name,v in p['files'].items():
  path=pathlib.Path(p['code_root'])/name;assert path.stat().st_size==v['bytes'] and sha(path)==v['sha256']
 original=read(b/'source-publication.json');assert original['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e';assert sha(b/'config.json')=='3e910b12c0ff52eac70e61566c6135e2ec188bb47ec64ad79524abbd7a657526'
 return p,source,original['code_root']
'''
STAGE_INIT=r'''
lock=(attempt/'operator-stage.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
root=attempt/('operator-'+inputs['commit'][:12]);root.mkdir(exist_ok=True)
pub=dict(commit=inputs['commit'],branch=inputs['branch'],code_root=str(root),files=inputs['files'],original_runtime_commit='501cf80478b77f68c3e13a96d4659e3da0971b2e')
dest=attempt/'operator-upload-intent.json'
if dest.exists():assert read(dest)==pub
else:atomic(dest,pub)
print(json.dumps(dict(status='IMMUTABLE_RESUME_CODE_UPLOAD_BOUND')))
'''
STAGE_FILE=r'''
lock=(attempt/'operator-stage.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pub=read(attempt/'operator-upload-intent.json');assert pub['commit']==inputs['commit'];name=inputs['name']
assert name in pub['files'] and pathlib.Path(name).name==name
data=zlib.decompress(base64.b64decode(inputs['data']));v=pub['files'][name];assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256'];p=pathlib.Path(pub['code_root'])/name
if p.exists():assert p.read_bytes()==data
else:p.write_bytes(data);p.chmod(0o444)
print(json.dumps(dict(status='RESUME_CODE_FILE_SHA_ACCEPTED',name=name)))
'''
STAGE=r'''
lock=(attempt/'operator-stage.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pub=read(attempt/'operator-upload-intent.json');assert pub['commit']==inputs['commit']
for name,v in pub['files'].items():
 p=pathlib.Path(pub['code_root'])/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
dest=attempt/'operator-publication.json'
if dest.exists():assert read(dest)==pub
else:atomic(dest,pub)
publication();print(json.dumps(dict(status='INDEPENDENT_EXACT_RESUME_OPERATOR_CODE_STAGED_NO_GPU_OR_HANDOFF',publication=ref(dest))))
'''
START=r'''
lock=(attempt/'operator-control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pub,source,code_root=publication();action=inputs['action'];controller=attempt/(action+'-controller.json')
if controller.exists():print(json.dumps(dict(status='EXISTING_CPU_ACTION_INTENT_INSPECT_STATUS_NO_DUPLICATE',controller=read(controller))));sys.exit(0)
base="import importlib.util,sys,pathlib,json,unittest; p=pathlib.Path(%r); spec=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.direct_hidden_resume',str(p)); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); "%(str(source))
if action=='qualify':
 script=base+"ds=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.deploy_direct_hidden_resume',%r); d=importlib.util.module_from_spec(ds); sys.modules[ds.name]=d; ds.loader.exec_module(d); s=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.test_direct_hidden_resume',%r); t=importlib.util.module_from_spec(s); s.loader.exec_module(t); suite=unittest.defaultTestLoader.loadTestsFromModule(t); result=unittest.TextTestRunner(verbosity=2).run(suite); assert result.wasSuccessful(); module.a.immutable(module.ATTEMPT/'CPU-qualification.json',dict(status='EXACT_JOINT_RESUME_CPU_FOCUSED_TESTS_PASSED',tests=result.testsRun,GPU_used=False,real_future_checkpoint_claimed=False)); print(json.dumps(dict(status='EXACT_RESUME_CPU_QUALIFIED',tests=result.testsRun)))"%(str(source.parent/'deploy_direct_hidden_resume.py'),str(source.parent/'test_direct_hidden_resume.py'))
else:
 assert action=='prepare';assert read(attempt/'CPU-qualification.json')['tests']==4
 q=command(['squeue','-j','743619','-h','-o','%i|%T']);assert not q.strip(),'Original writer still active; preparation waits for actual terminal and saved interrupted state'
 assert (b/'interrupted.json').exists(),'Actual interrupted receipt required, no invented future CP'
 script=base+"print(json.dumps(module.prepare(%r,%r)))"%(str(source),sha(source))
env=dict(os.environ,QWEN7B_CODE_ROOT=code_root,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(attempt/(action+'.log')).open('ab',0)
args=['bash',code_root+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u','-c',script];p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env);atomic(controller,dict(pid=p.pid,args=args,commit=pub['commit'],CPU_only=True,time=time.time()));print(json.dumps(dict(status='FINITE_EXACT_RESUME_CPU_ACTION_STARTED',action=action,pid=p.pid)))
'''
SUBMIT=r'''
lock=(attempt/'operator-control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
training=(b/'joint-training.lock').open('a+');fcntl.flock(training,fcntl.LOCK_EX|fcntl.LOCK_NB)
pub,source,code_root=publication();p=read(attempt/'preparation.json')
assert p['status']=='ACTUAL_INTERRUPTED_JOINT_STATE_VERIFIED_CONTINUATION_PREPARED_NOT_SUBMITTED'
assert not (attempt/'failure.json').exists() and not (b/'failure.json').exists()
assert not command(['squeue','-j','743619','-h','-o','%i|%T']).strip()
assert read(b/'latest.json')['sha256']==p['source_checkpoint']['sha256'] and read(b/'progress.json')['cursor']==p['source_cursor']
assert read(b/'update-intent.json')['next_cursor']<=p['source_cursor'] and read(b/'interrupted.json')['cursor']==p['source_cursor']
script=pathlib.Path(p['sbatch']['path']);assert ref(script)==p['sbatch']
if (attempt/'job.id').exists():print(json.dumps(dict(status='EXISTING_CONTINUATION_NO_DUPLICATE',job_id=(attempt/'job.id').read_text().strip())));sys.exit(0)
name='q7b-direct-count938-r1';q=command(['squeue','-u','mou176','-h','-o','%i|%j|%T']);assert not any('|'+name+'|' in x for x in q.splitlines())
assert not (attempt/'submit-intent.json').exists(),'Unknown prior submission; inspect job identity before another attempt'
atomic(attempt/'submit-intent.json',dict(job_name=name,sbatch=p['sbatch'],source_checkpoint=p['source_checkpoint'],operator_commit=pub['commit'],time=time.time()))
out=command(['sbatch','--hold','--parsable',str(script)]);jid=out.split(';')[0];assert jid.isdigit();(attempt/'job.id').write_text(jid+'\n')
identity=command(['scontrol','show','job','-o',jid]);assert 'JobName='+name in identity and 'WorkDir='+str(b) in identity and 'Account=mou176' in identity and 'TimeLimit=03:00:00' in identity
atomic(attempt/'submission.json',dict(job_id=jid,identity=identity,source_checkpoint=p['source_checkpoint'],source_cursor=p['source_cursor'],remaining_updates=p['remaining_updates'],dependency=None,operator_commit=pub['commit'],runtime_commit=p['runtime_commit'],time=time.time()))
command(['scontrol','release',jid]);print(json.dumps(dict(status='ONE_EXACT_JOINT_CONTINUATION_SUBMITTED',job_id=jid,source_cursor=p['source_cursor'],remaining_updates=p['remaining_updates'],normal=True,dependency=None)))
'''
STATUS=r'''
v=dict(status='INDEPENDENT_EXACT_JOINT_CONTINUATION_ACTUAL_STATE',time=time.time(),original_job_id='743619')
for name in ('operator-publication.json','qualify-controller.json','CPU-qualification.json','prepare-controller.json','preparation.json','submit-intent.json','submission.json','controller.json','next-update-replay.json','first-cp-independent-audit.json','handoff-complete.json','failure.json'):
 p=attempt/name
 if p.exists():
  x=read(p)
  if name=='operator-publication.json':x={k:x[k] for k in ('commit','code_root','original_runtime_commit')}
  v['continuation_'+name[:-5]]=x
for name in ('progress.json','latest.json','interrupted.json','complete.json'):
 if (b/name).exists():v[name[:-5]]=read(b/name)
jid=(attempt/'job.id').read_text().strip() if (attempt/'job.id').exists() else '743619';assert jid.isdigit();v['job_id']=jid;v['squeue']=command(['squeue','-j',jid,'-h','-o','%i|%j|%T|%M|%R|%N']);v['sacct']=command(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList'])
if (b/'failure.json').exists():v['failure']=read(b/'failure.json')
for cursor in (256,512,768,938):
 p=b/'evaluations'/('cursor-%06d'%cursor)/'complete.json'
 if p.exists():v['evaluation_'+str(cursor)]=read(p)
for action in ('qualify','prepare'):
 log=attempt/(action+'.log')
 if log.exists() and not (attempt/('CPU-qualification.json' if action=='qualify' else 'preparation.json')).exists():v[action+'_log_tail']=log.read_text()[-4500:]
logs=sorted((b/'logs').glob('resume-*.log'))
if logs:v['gpu_log_tail']=logs[-1].read_text()[-2200:]
print(json.dumps(v))
'''


def remote_source(values,action):
    encoded=base64.b64encode(zlib.compress(json.dumps(values,sort_keys=True).encode())).decode()
    return 'import base64,json,zlib,sys\ninputs=json.loads(zlib.decompress(base64.b64decode('+repr(encoded)+')))\n'+COMMON+dict(stage=STAGE,stage_init=STAGE_INIT,stage_file=STAGE_FILE,qualify=START,prepare=START,submit=SUBMIT,status=STATUS)[action]


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--action',choices=('stage','qualify','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');args=p.parse_args()
    if args.action=='submit' and not args.submit_authorized:p.error('Existing parent continuation authorization and --submit-authorized required')
    repo=Path(__file__).resolve().parents[3];spec=importlib.util.spec_from_file_location('resume_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    values=dict(base=str(r.BASE),commit=args.commit,branch=BRANCH,action=args.action)
    def call(action,packet):
        command='python3 -c '+shlex.quote(remote_source(packet,action))
        if len(command.encode())>=16000:raise ValueError('Control command exceeds16KB')
        result=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=50)
        if result.returncode:raise RuntimeError('Outcome unconfirmed; inspect original and continuation identity first: '+str(result.stderr[-2200:]))
        return json.loads(result.stdout.splitlines()[-1])
    if args.action=='stage':
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]
        if remote!=args.commit:raise ValueError('Exact continuation source must be published first')
        files={}
        for name in OWN:
            rel='reproduction/operations/qwen7b_warmup/'+name;data=subprocess.check_output(['git','show',args.commit+':'+rel],cwd=repo)
            if data!=(repo/rel).read_bytes():raise ValueError('Unpublished continuation source '+rel)
            files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(zlib.compress(data)).decode())
        manifest={k:{x:v[x] for x in ('bytes','sha256')} for k,v in files.items()}
        call('stage_init',dict(values,files=manifest))
        for name,item in files.items():call('stage_file',dict(values,name=name,data=item['data']))
    data=call(args.action,values);print(json.dumps(data))
    path=args.control_root/'outputs/qwen7b-warmup-20261019/reports'/('direct-hidden-resume-'+args.action+'-latest.json');path.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
