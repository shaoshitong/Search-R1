"""Isolated joint runner operator; submit requires a later explicit parent GO."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile,zlib
from pathlib import Path
from . import direct_hidden_count_train as r,deploy_direct_hidden_count_prepare as prior
BRANCH='codex/direct-hidden-count-runner-20261021'
REFERENCE='28c38143fcf125c5c0035a97b78af11e13a50423'
OWN=('direct_hidden_count_train.py','deploy_direct_hidden_count_train.py','test_direct_hidden_count_train.py',
     'direct_hidden_checkpoint.py','test_direct_hidden_checkpoint.py')
COMMON,INIT,CHUNK,EXTRACT=prior.COMMON,prior.INIT,prior.CHUNK,prior.EXTRACT
PREPARE=r'''
h=lock();v,root=pub()
if (b/'preparation.json').exists():
 p=read(b/'preparation.json');verify(p['config']);verify(p['sbatch']);print(json.dumps(p));sys.exit(0)
if (b/'cpu-controller.json').exists():raise RuntimeError('Existing CPU prepare intent; inspect status before another attempt')
code="import unittest,sys,runpy; s=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('reproduction.operations.qwen7b_warmup.'+n) for n in ['test_direct_hidden_count','test_direct_hidden_checkpoint','test_direct_hidden_count_train']); result=unittest.TextTestRunner(verbosity=2).run(s); assert result.wasSuccessful(); sys.argv=['direct_hidden_count_train','--action','prepare']; runpy.run_module('reproduction.operations.qwen7b_warmup.direct_hidden_count_train',run_name='__main__')"
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(b/'cpu-prepare.log').open('ab',0)
args=['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-c',code];p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env);atomic(b/'cpu-controller.json',dict(pid=p.pid,args=args,commit=v['commit'],CPU_only=True,time=time.time()));print(json.dumps(dict(status='JOINT_PRODUCTION_ENTRY_CPU_TESTS_AND_METADATA_STARTED_NO_GPU',pid=p.pid)))
'''
SUBMIT="training_guard=(b/'joint-training.lock').open('a+');fcntl.flock(training_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"+prior.prior.prior.SUBMIT.replace('q7b-peer8192-within1','q7b-direct-count938').replace('ONE_FROZEN_CALIBRATION_GPU_JOB_SUBMITTED','ONE_JOINT_HIDDEN_COUNT938_GPU_JOB_SUBMITTED')
STATUS=r'''
v=dict(status='JOINT_HIDDEN_COUNT938_ACTUAL_STATE',time=time.time())
for name in ('source-publication.json','cpu-controller.json','preparation.json','submission.json','controller.json','progress.json','failure.json','first-cp-independent-audit.json','first-cp-audit-failure.json','next-update-replay.json','latest.json','interrupted.json','complete.json'):
 p=b/name
 if p.exists():
  x=read(p)
  if name=='source-publication.json':x={k:x[k] for k in ('commit','code_root')}
  v[name[:-5]]=x
if (b/'job.id').exists():
 jid=(b/'job.id').read_text().strip();assert jid.isdigit();v['job_id']=jid;v['squeue']=command(['squeue','-j',jid,'-h','-o','%i|%j|%T|%M|%R|%N']);v['sacct']=command(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList'])
for cursor in (256,512,768,938):
 p=b/'evaluations'/('cursor-%06d'%cursor)/'complete.json'
 if p.exists():v['evaluation_'+str(cursor)]=read(p)
if (b/'cpu-prepare.log').exists() and not (b/'preparation.json').exists():v['cpu_log_tail']=(b/'cpu-prepare.log').read_text()[-5000:]
logs=sorted((b/'logs').glob('train-*.log')) if (b/'logs').exists() else []
if logs:v['gpu_log_tail']=logs[-1].read_text()[-2000:]
print(json.dumps(v))
'''

def remote_source(values,action):
    # The growing closure's manifest can exceed the16KB transport ceiling.
    # Compress structured control metadata, never payload/model/credential data.
    encoded=base64.b64encode(zlib.compress(json.dumps(values,sort_keys=True).encode())).decode()
    return 'import base64,json,zlib\ninputs=json.loads(zlib.decompress(base64.b64decode('+repr(encoded)+')))\n'+COMMON+dict(init=INIT,chunk=CHUNK,extract=EXTRACT,prepare=PREPARE,submit=SUBMIT,status=STATUS)[action]

def bundle(repo,commit):
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]
    if remote!=commit:raise ValueError('Exact production entry source must be published first')
    data,files,origins=prior.bundle(repo,REFERENCE);payloads={}
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for m in archive.getmembers():payloads[m.name]=archive.extractfile(m).read()
    for short in OWN:
        name=r.a.PKG+short;data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
        if data!=(repo/name).read_bytes():raise ValueError('Unpublished joint production entry '+name)
        payloads[name]=data;origins['overrides'][name]=commit
    output=io.BytesIO();files={}
    with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name,data in sorted(payloads.items()):
                files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o444;m.mtime=0;archive.addfile(m,io.BytesIO(data))
    return output.getvalue(),files,origins

def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');args=p.parse_args()
    if args.action=='submit' and not args.submit_authorized:p.error('Explicit parent GO plus --submit-authorized required; prepare alone is not GO')
    spec=importlib.util.spec_from_file_location('joint_direct_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    values=dict(base=str(r.TARGET),target=str(r.TARGET),pin=r.a.D8,callback_commit=r.a.CALLBACK,reference=prior.REFERENCE)
    def call(action,extra=None):
        v=dict(values);v.update(extra or {});command='python3 -c '+shlex.quote(remote_source(v,action))
        if len(command.encode())>=16000:raise ValueError('Control command exceeds16KB')
        result=ctl.run(command,account='mou176',source_address=ip,timeout=55)
        lines=[json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')]
        if result.returncode or not lines:raise RuntimeError('Unconfirmed control outcome; inspect own identity before retry: '+result.stderr[-1500:])
        return lines[-1]
    if args.action=='stage':
        data,files,origins=bundle(Path(__file__).resolve().parents[3],args.commit);chunks=[data[i:i+4096] for i in range(0,len(data),4096)]
        call('init',dict(commit=args.commit,remote_branch=BRANCH,bundle_sha256=hashlib.sha256(data).hexdigest(),bundle_bytes=len(data),chunks=len(chunks),files=files,file_origins=origins))
        for i,part in enumerate(chunks):call('chunk',dict(commit=args.commit,index=i,data=base64.b64encode(part).decode(),sha256=hashlib.sha256(part).hexdigest()))
        result=call('extract',dict(commit=args.commit))
    else:result=call(args.action)
    print(json.dumps(result));path=args.control_root/'outputs/qwen7b-warmup-20261019/reports'/('direct-hidden-count-production-'+args.action+'-latest.json');path.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
