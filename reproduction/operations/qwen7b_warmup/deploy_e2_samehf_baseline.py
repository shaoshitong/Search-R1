"""Isolated sealed-E2 same-HF validation deployer; no calibration-grid mutation."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import deploy_peer_g10_calibration as transport,score_e2_samehf_baseline as e2
FILES=('score_e2_samehf_baseline.py','deploy_e2_samehf_baseline.py','test_e2_samehf_baseline.py','score_peer_g10_calibration.py','deploy_peer_g10_calibration.py')
BRANCH=transport.BRANCH
PREPARE=transport.PREPARE.replace('score_peer_g10_calibration.py','score_e2_samehf_baseline.py')
SUBMIT=transport.SUBMIT.replace('q7b-peer8192-frozen-cal',e2.JOB)
STATUS=r'''
v=dict(status='SEALED_E2_SAME_HF_ACTUAL_STATE',observed_epoch=time.time())
for name in ('source-publication.json','preparation.json','controller.json','progress.json','failure.json'):
 p=b/name
 if p.exists():
  x=read(p)
  if name=='source-publication.json':x={k:x[k] for k in ('commit','code_root','pinned_runtime_commit')}
  v[name[:-5]]=x
if (b/'job.id').exists():
 jid=(b/'job.id').read_text().strip();assert jid.isdigit();v['job_id']=jid;v['squeue']=command(['squeue','-j',jid,'-h','-o','%i|%j|%T|%M|%R|%N']);v['sacct']=command(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList']);v['start_estimate']=command(['squeue','--start','-j',jid,'-h','-o','%i|%T|%S|%R'])
p=b/'complete.json'
if p.exists():v['complete']=read(p);v['complete_ref']=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
if (b/'cpu-prepare.log').exists() and not (b/'preparation.json').exists():v['cpu_log_tail']=(b/'cpu-prepare.log').read_text()[-2500:]
logs=sorted((b/'logs').glob('scoring-*.log')) if (b/'logs').exists() else []
if logs:v['gpu_log_tail']=logs[-1].read_text()[-1500:]
print(json.dumps(v))
'''
def remote_source(v,action):return 'inputs='+repr(v)+'\n'+transport.COMMON+dict(init=transport.INIT,chunk=transport.CHUNK,extract=transport.EXTRACT,prepare=PREPARE,submit=SUBMIT,status=STATUS)[action]
def bundle(repo,commit):
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]
 if commit!=remote:raise ValueError('Publish E2 source before staging')
 data=io.BytesIO();files={}
 with gzip.GzipFile(fileobj=data,mode='wb',filename='',mtime=0) as zipped:
  with tarfile.open(fileobj=zipped,mode='w') as archive:
   for name in FILES:
    rel='reproduction/operations/qwen7b_warmup/'+name;payload=subprocess.check_output(['git','show',commit+':'+rel],cwd=repo)
    if payload!=(repo/rel).read_bytes():raise ValueError('Unpublished E2 dependency '+name)
    files[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest());m=tarfile.TarInfo(name);m.size=len(payload);m.mode=0o444;m.mtime=0;archive.addfile(m,io.BytesIO(payload))
 return data.getvalue(),files

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit' and not a.submit_authorized:p.error('--submit-authorized required')
 spec=importlib.util.spec_from_file_location('E2_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();base=dict(base=str(e2.TARGET),target=str(e2.TARGET),train=str(e2.s.TRAIN),pin=e2.s.PIN)
 def call(action,extra=None):
  own=dict(base);own.update(extra or {});command='python3 -c '+shlex.quote(remote_source(own,action))
  if len(command.encode())>=16000:raise ValueError('Control payload exceeds16KB')
  r=control.run(command,account='mou176',source_address=address,timeout=55);rows=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
  if r.returncode or not rows:raise RuntimeError('E2 outcome unconfirmed, inspect before retry: '+r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-2500:])
  return rows[-1]
 if a.action=='stage':
  data,files=bundle(Path(__file__).resolve().parents[3],a.commit);chunks=[data[i:i+4096] for i in range(0,len(data),4096)];v=dict(commit=a.commit,remote_branch=BRANCH,bundle_sha256=hashlib.sha256(data).hexdigest(),bundle_bytes=len(data),chunks=len(chunks),files=files);call('init',v)
  for i,chunk in enumerate(chunks):call('chunk',dict(commit=a.commit,index=i,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest()))
  result=call('extract',v)
 else:result=call(a.action)
 print(json.dumps(result,sort_keys=True));out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('E2-same-HF-'+a.action+'-latest.json');out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
