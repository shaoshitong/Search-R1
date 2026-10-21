"""One guarded normal/acd_u H100 capacity job, no dependencies or actor writes."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
REPO=pathlib.Path(__file__).resolve().parents[3];CONTROL=REPO.parents[1]
NAME='reproduction/operations/qwen7b_warmup/direct_count_capacity.py'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-head-capacity1536-v1'
PRODUCTION='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2'
COMMON='''import pathlib,json,hashlib,os,fcntl,time,subprocess,base64
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);lock=(b/'operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def runtime():return read(pathlib.Path(v['production'])/'source-publication.json')['code_root']
def command(argv):
 p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit'and not a.submit_authorized:p.error('Parent GO plus explicit --submit-authorized required')
 spec=importlib.util.spec_from_file_location('capacity_diagnostic_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**values):
  values.update(base=BASE,production=PRODUCTION);cmd='python3 -c '+shlex.quote('v='+repr(values)+'\n'+COMMON+body)
  if len(cmd.encode())>=16000:raise ValueError('Bounded control commands must stay under16KB')
  r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError('Unconfirmed action: inspect existing intent/status before retry: '+r.stderr.decode()[-1600:]+r.stdout.decode()[-2000:])
  return json.loads(next(x for x in reversed(r.stdout.decode().splitlines())if x.startswith('{')))
 if a.action=='stage':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=REPO).decode().split()[0]
  if commit!=remote:raise ValueError('Publish current runtime and operator first')
  raw=subprocess.check_output(['git','show',commit+':'+NAME],cwd=REPO)
  if raw!=(REPO/NAME).read_bytes():raise ValueError('Unpublished capacity source refused')
  for i,start in enumerate(range(0,len(raw),4096)):
   part=raw[start:start+4096];call("root=b/('code-'+v['commit'][:12]);root.mkdir(exist_ok=True);p=root/('part-%04d'%v['i']);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha']\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(part=v['i'])))",commit=commit,i=i,data=base64.b64encode(part).decode(),sha=hashlib.sha256(part).hexdigest())
  result=call("""root=b/('code-'+v['commit'][:12]);raw=b''.join((root/('part-%04d'%i)).read_bytes()for i in range(v['parts']));assert hashlib.sha256(raw).hexdigest()==v['sha'];source=root/'capacity.py';compile(raw,str(source),'exec')
if source.exists():assert source.read_bytes()==raw
else:source.write_bytes(raw);source.chmod(0o444)
pub=dict(commit=v['commit'],remote_sha=v['commit'],source=proof(source));p=b/'source-publication.json'
if p.exists():assert read(p)==pub
else:p.write_text(json.dumps(pub,sort_keys=True)+'\\n')
rt=runtime();env=dict(os.environ,QWEN7B_CODE_ROOT=rt,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',rt+'/reproduction/operations/qwen7b_warmup/runtime.sh',str(source),'--action','self-test'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=env);assert r.returncode==0,r.stderr
print(json.dumps(dict(status='FROZEN_HEAD_CAPACITY_CODE_ONLY_STAGED',publication=proof(p),fixture=r.stdout.strip())))""",commit=commit,parts=(len(raw)+4095)//4096,sha=hashlib.sha256(raw).hexdigest())
 elif a.action=='prepare':
  result=call("""if (b/'preparation.json').exists():print(json.dumps(read(b/'preparation.json')))
else:
 pub=read(b/'source-publication.json');assert proof(pub['source']['path'])==pub['source'];rt=runtime();env=dict(os.environ,QWEN7B_CODE_ROOT=rt,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 r=subprocess.run(['bash',rt+'/reproduction/operations/qwen7b_warmup/runtime.sh',pub['source']['path'],'--action','prepare'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=env,timeout=48)
 assert r.returncode==0,r.stderr[-3000:];print(json.dumps(read(b/'preparation.json')))""")
 elif a.action=='submit':
  result=call("""prep=read(b/'preparation.json');assert prep['fit']==512 and prep['holdout']==1024;assert proof(prep['config']['path'])==prep['config'];assert proof(prep['sbatch']['path'])==prep['sbatch']
if (b/'job.id').exists():print(json.dumps(dict(status='EXISTING_JOB_PRESERVED',job_id=(b/'job.id').read_text().strip())))
else:
 assert not (b/'submit-intent.json').exists(),'Unknown submission outcome: reconcile first, never duplicate';q=command(['squeue','-h','-u','mou176','-n','q7b-headcap1536','-o','%i|%j|%T|%N']);assert q['returncode']==0 and not q['stdout'].strip(),q
 with (b/'submit-intent.json').open('x')as f:json.dump(dict(config=prep['config'],sbatch=prep['sbatch'],epoch=time.time()),f);f.flush();os.fsync(f.fileno())
 r=command(['sbatch','--parsable',prep['sbatch']['path']]);assert r['returncode']==0,r
 jid=r['stdout'].strip().split(';')[0];assert jid.isdigit(),r
 (b/'job.id').write_text(jid+'\\n');receipt=dict(status='ONE_NORMAL_H100_HEAD_CAPACITY_DIAGNOSTIC_SUBMITTED',job_id=jid,resources=dict(GPU=1,CPU=8,memory='64G',minutes=30,partition='acd_u',QoS='normal',dependencies=None),config=prep['config'],epoch=time.time());(b/'submission.json').write_text(json.dumps(receipt)+'\\n');print(json.dumps(receipt))""")
 else:
  result=call("""out=dict(epoch=time.time())
for name in ['source-publication.json','preparation.json','submission.json','progress.json','cache-complete.json','failure.json','interrupted.json']:
 if (b/name).exists():out[name[:-5]]=read(b/name)
for d in sorted(b.glob('lr-*')):
 rec={}
 for name in ['first-update-replay.json','latest.json']:
  if (d/name).exists():rec[name[:-5]]=read(d/name)
 if (d/'complete.json').exists():
  v2=read(d/'complete.json');rec['complete']=dict(status=v2['status'],learning_rate=v2['learning_rate'],checkpoint=v2['checkpoint'],first_six=v2['first_six'],last_evaluation=v2['evaluations'][-1])
 if rec:out[d.name]=rec
if (b/'complete.json').exists():out['complete']=dict(receipt=proof(b/'complete.json'),legacy_unchanged=read(b/'complete.json')['legacy_unchanged'],max_GPU_allocated_bytes=read(b/'complete.json')['max_GPU_allocated_bytes'])
if (b/'job.id').exists():
 jid=(b/'job.id').read_text().strip();assert jid.isdigit();out['job_id']=jid;out['squeue']=command(['squeue','-h','-j',jid,'-o','%i|%j|%T|%M|%R|%N']);out['sacct']=command(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])
print(json.dumps(out))""")
 out=REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/direct-count-capacity';out.mkdir(exist_ok=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
