"""External CPU audit using the actual separately recorded continuation snapshot."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,zlib
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
SOURCE_COMMIT='0be87366db66cbd6705e0806e435a680323e9b97'
RUNTIME_COMMIT='501cf80478b77f68c3e13a96d4659e3da0971b2e'
NAME='reproduction/operations/qwen7b_warmup/audit_direct_hidden_count.py'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2'
p=argparse.ArgumentParser();p.add_argument('--cursor',type=int,choices=(768,938),required=True);p.add_argument('--job-id',required=True);args=p.parse_args()
state=json.loads((OUT/'actual-resume-observer/latest-good.json').read_text())
if not args.job_id.isdigit() or args.job_id=='743619' or state['job_id']!=args.job_id:raise ValueError('Actual separately recorded continuation job ID required')
if state['original_job_id']!='743619' or state['continuation_operator-publication']['commit']!='31e930ce8327078612c3ee9dd98aec2188a78826':raise ValueError('Qualified exact continuation lineage differs')
if state.get('continuation_handoff-complete',{}).get('status')!='CONTINUATION_FIRST_NEW_UPDATE_REPLAY_AND_INDEPENDENT_FULLCP_ACCEPTED':raise ValueError('Actual accepted continuation handoff required')
if 'evaluation_'+str(args.cursor) not in state:raise RuntimeError('Wait for actual checkpoint-first fixed evaluation complete')
main_jobs=[line.split('|') for line in state.get('sacct','').splitlines() if line.split('|',1)[0]==args.job_id]
if args.cursor==938 and (len(main_jobs)!=1 or main_jobs[0][2:4]!=['COMPLETED','0:0']):raise RuntimeError('Final audit waits for this continuation job normal terminal Slurm state')
payload=subprocess.check_output(['git','show',SOURCE_COMMIT+':'+NAME],cwd=ROOT)
if payload!=(ROOT/NAME).read_bytes():raise ValueError('External auditor differs from published source')
values=dict(base=BASE,cursor=args.cursor,source_commit=SOURCE_COMMIT,runtime_commit=RUNTIME_COMMIT,
    source_sha256=hashlib.sha256(payload).hexdigest(),payload=base64.b64encode(zlib.compress(payload)).decode())
stage='''import pathlib,json,base64,zlib,hashlib,fcntl,os
b=pathlib.Path(values['base']);p=b.parent/('direct-hidden-count-independent-audit-'+values['source_commit'][:12]);p.mkdir(exist_ok=True)
lock=(p/('cursor-%06d.lock'%values['cursor'])).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pub=json.loads((b/'source-publication.json').read_text());assert pub['commit']==values['runtime_commit']
assert (b/'evaluations'/('cursor-%06d'%values['cursor'])/'complete.json').is_file()
data=zlib.decompress(base64.b64decode(values['payload']));assert hashlib.sha256(data).hexdigest()==values['source_sha256']
source=p/'audit_direct_hidden_count.py'
if source.exists():assert source.read_bytes()==data
else:
 tmp=source.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(source);source.chmod(0o444)
print(json.dumps(dict(code_root=pub['code_root'],source=str(source),source_sha256=values['source_sha256'])))'''
spec=importlib.util.spec_from_file_location('joint_milestone_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
def call(command,timeout=90):
    if len(command.encode())>=16000:raise ValueError('Control command exceeds16KB')
    r=ctl.run(command,account='mou176',source_address=ip,timeout=timeout)
    if r.returncode:raise RuntimeError(str(r.stderr[-2200:]))
    return json.loads(r.stdout.splitlines()[-1])
info=call('python3 -c '+shlex.quote('values='+repr(values)+'\n'+stage),45)
worker='''import importlib.util,pathlib,json,hashlib,fcntl
b=pathlib.Path(values['base']);source=pathlib.Path(values['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==values['source_sha256']
lock=(source.parent/('cursor-%06d.lock'%values['cursor'])).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
audit=b/'evaluations'/('cursor-%06d'%values['cursor'])/'independent-audit.json'
if audit.exists():
 result=json.loads(audit.read_text());assert result['config']['sha256']==hashlib.sha256((b/'config.json').read_bytes()).hexdigest();assert result['cursor']==values['cursor']
 for item in list(result['predictions'].values())+[result['evaluation_receipt'],result['config']]:
  file=pathlib.Path(item['path']);assert file.stat().st_size==item['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
else:
 spec=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.external_joint_audit',str(source));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);result=module.audit(b,values['cursor'])
print(json.dumps(dict(status=result['status'],cursor=result['cursor'],metrics=result['metrics'],checkpoint=result['checkpoint'],audit=dict(path=str(audit),bytes=audit.stat().st_size,sha256=hashlib.sha256(audit.read_bytes()).hexdigest()))))'''
v=dict(base=BASE,cursor=args.cursor,**info)
command='export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 QWEN7B_CODE_ROOT='+shlex.quote(info['code_root'])+'; bash '+shlex.quote(info['code_root']+'/reproduction/operations/qwen7b_warmup/runtime.sh')+' -u -c '+shlex.quote('values='+repr(v)+'\n'+worker)
result=call(command,95);result['published_external_auditor_commit']=SOURCE_COMMIT;result['actual_continuation_job_id']=args.job_id;result['original_job_id']='743619'
(OUT/('actual-cursor-%06d-independent-audit.json'%args.cursor)).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
