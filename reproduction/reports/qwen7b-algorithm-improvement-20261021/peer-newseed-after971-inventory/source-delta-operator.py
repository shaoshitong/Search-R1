"""Bounded source-side CPU audit operator. No producer writes or GPU work."""
import argparse, hashlib, importlib.util, json, pathlib, shlex, subprocess, zlib
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent
PREFIX='reproduction/operations/qwen7b_warmup/'


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('launch','status','fetch-plan','alias-plan','worker'),required=True);p.add_argument('--commit',required=True);p.add_argument('--branch',default='codex/peer-donor-after971-20261021');a=p.parse_args()
    source=json.loads((HERE/'actual-source-after971.json').read_text());h=source['delta_records_sha256']
    root='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/source-proof-after971-'+h[:12]
    req=dict(action=a.action,root=root,commit=a.commit)
    if a.action=='launch':
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+a.branch],cwd=R,text=True).split()[0]
        if remote!=a.commit:raise ValueError('Audit source must be published at its exact immutable branch')
        names=['peer_newseed_delta.py','peer_newseed_freeze_after466.py','peer_newseed_audit_reference.py','peer_newseed_audit_frozen_delta.py']
        req['code']={PREFIX+n:subprocess.check_output(['git','show',a.commit+':'+PREFIX+n],cwd=R).decode()for n in names}
        ex=(R.parents[1]/'outputs/qwen7b-warmup-20261019/after971-evaluation-identity-exclusions.json').read_bytes().strip()+b'\n'
        if len(ex)!=848051 or hashlib.sha256(ex).hexdigest()!='585893c9960fcb6b1934887ebb3034c7da1f479197ddd9e30201569ebdb5fdfc':raise ValueError('Actual frozen evaluation identity metadata differs')
        req.update(reference=source,exclusions=ex.decode())
    worker=r'''import pathlib,json,sys,hashlib,fcntl,os,subprocess,time,zlib
p=json.loads(zlib.decompress(sys.stdin.buffer.read()));root=pathlib.Path(p['root'])
def load(path):
 raw=path.read_bytes();return json.loads(raw),dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
if p['action']=='worker':
 proc=pathlib.Path('/proc/3532221');out=dict(host=os.uname().nodename,pid=3532221,exists=proc.exists(),epoch=time.time())
 if proc.exists():
  st=(proc/'stat').read_text().rsplit(')',1)[1].split();cmd=(proc/'cmdline').read_bytes();out.update(birth_ticks=int(st[19]),state=st[0],cmdline_sha256=hashlib.sha256(cmd).hexdigest(),sampling_worker_command=(b'prefix_selection200' in cmd or b'pool' in cmd or b'sample' in cmd),source_run_id_present=b'70188b8e50fd' in cmd)
 print(json.dumps(out));sys.exit(0)
if p['action']=='alias-plan':
 complete,cref=load(root/'complete.json');assert complete['status']=='FINITE_DONOR_DELTA_FRESH_SHA_CAUSAL_OWN_G5_AUDIT_AND_FROZEN_PLAN_COMPLETE'
 plan,ref=load(pathlib.Path(complete['plan']['path']));assert ref==complete['plan'] and plan['batches']==138 and plan['questions']==17664
 dest=root.parent/'metadata-plans/after971-51057f377942/audited138-plan.json';dest.parent.mkdir(parents=True,exist_ok=True);blob=pathlib.Path(ref['path']).read_bytes()
 if dest.exists():assert dest.read_bytes()==blob
 else:
  with dest.open('xb')as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
 assert load(dest)[0]==plan;print(json.dumps(dict(status='QUALIFIED_FROZEN_PLAN_BYTE_IDENTICAL_ALIAS_ACCEPTED',source_complete=cref,source_plan=ref,plan=load(dest)[1])));sys.exit(0)
if p['action']=='launch':
 root.mkdir(parents=True,exist_ok=True)
 with(root/'launch.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(root/'launch-intent.json').exists():
   old,_=load(root/'launch-intent.json');assert old['commit']==p['commit'];print(json.dumps(dict(status='EXISTING_OWN_LAUNCH_PRESERVED',root=str(root),intent=old)));sys.exit(0)
  code=root/('code-'+p['commit'][:12]);code.mkdir(exist_ok=True)
  for name,text in p['code'].items():
   dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True)
   for folder in (dest.parent,dest.parent.parent,dest.parent.parent.parent):
    init=folder/'__init__.py'
    if not init.exists():init.write_bytes(b'')
   if dest.exists():assert dest.read_text()==text
   else:dest.write_text(text)
  ref=root/'finite-reference.json';ex=root/'evaluation-identity-exclusions.json'
  rb=json.dumps(p['reference'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n';eb=p['exclusions'].encode()
  for dest,blob in((ref,rb),(ex,eb)):
   if dest.exists():assert dest.read_bytes()==blob
   else:dest.write_bytes(blob)
  intent=dict(commit=p['commit'],source_reference=load(ref)[1],evaluation_exclusions=load(ex)[1],code={k:hashlib.sha256(v.encode()).hexdigest()for k,v in p['code'].items()},epoch=time.time())
  (root/'launch-intent.json').write_text(json.dumps(intent))
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  log=(root/'controller.log').open('ab',buffering=0)
  child=subprocess.Popen([sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.peer_newseed_audit_frozen_delta','--reference',str(ref),'--reference-sha256',intent['source_reference']['sha256'],'--exclusions',str(ex),'--exclusions-sha256',intent['evaluation_exclusions']['sha256'],'--output',str(root)],cwd=code,env=env,stdout=log,stderr=log,start_new_session=True)
  st=pathlib.Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split();receipt=dict(status='FINITE_DONOR_DELTA_CPU_AUDIT_LAUNCHED',root=str(root),pid=child.pid,birth_ticks=int(st[19]),commit=p['commit'],source_reference=intent['source_reference'],epoch=time.time());(root/'launch.json').write_text(json.dumps(receipt));print(json.dumps(receipt));sys.exit(0)
out=dict(root=str(root),epoch=time.time(),files={})
for name in('launch.json','progress.json','complete.json','failure.json'):
 path=root/name
 if path.exists():v,r=load(path);out['files'][name]=dict(ref=r,value=v)
if(root/'controller.log').exists():out['log_tail']=(root/'controller.log').read_text()[-1800:]
launch=out['files'].get('launch.json',{}).get('value',{});proc=pathlib.Path('/proc/%s'%launch.get('pid',0));out['controller_same_birth']=False
if(proc/'stat').exists():
 st=(proc/'stat').read_text().rsplit(')',1)[1].split();out['controller_same_birth']=int(st[19])==launch.get('birth_ticks');out['controller_state']=st[0]
if p['action']=='fetch-plan':
 assert'complete.json'in out['files'];complete=out['files']['complete.json']['value'];plan,r=load(pathlib.Path(complete['plan']['path']));assert r==complete['plan'];out['frozen_plan']=plan
print(json.dumps(out))'''
    spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    host='h100-5'if a.action=='worker'else'h100-10';cmd='python3 -c '+shlex.quote(worker);assert len(cmd.encode())<16000
    wire=zlib.compress(json.dumps(req).encode(),6)
    r=subprocess.run(ctl.route(host,cmd),input=wire,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1800:])
    v=json.loads(r.stdout.decode().splitlines()[-1]);out=HERE/('actual-'+a.action+'.json');out.write_text(json.dumps(v,indent=2)+'\n')
    if a.action=='fetch-plan':(HERE/'frozen-audited-delta-plan.json').write_text(json.dumps(v['frozen_plan'],indent=2)+'\n');v.pop('frozen_plan')
    print(json.dumps(v))

if __name__=='__main__':main()
