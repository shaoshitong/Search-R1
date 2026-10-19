"""Independent fresh source-bound target readback. No model body leaves A800."""
import sys,subprocess,json,pathlib,shlex,hashlib,time
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
D=pathlib.Path('outputs/grounding-eval-20261019');expected=json.loads((D/'easy-a800-asset-job.json').read_text())
code=r'''import sys,json,pathlib,hashlib,time,os,subprocess
j=json.load(sys.stdin);root=pathlib.Path(j['output']);start=time.time();s=json.loads((root/'asset-status.json').read_text());complete=json.loads((root/'asset-migration-complete.json').read_text());assert s['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' and complete['status']==s['status'];assert json.loads((root/'asset-binding.json').read_text())==j
actual={};processed_bytes=0;last_progress=start;next_progress=1<<30
for n,v in j['files'].items():
 p=root/n;a=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):
   h.update(b);processed_bytes+=len(b);now=time.time()
   if processed_bytes>=next_progress or now-last_progress>=5:
    print(json.dumps({'phase':'INDEPENDENT_SOURCE_SHA_READBACK','processed_bytes':processed_bytes,'at':now}),flush=True);last_progress=now;next_progress=processed_bytes+(1<<30)
 z=p.stat();assert(a.st_size,a.st_ino,a.st_mtime_ns)==(z.st_size,z.st_ino,z.st_mtime_ns);assert a.st_size==v['size'] and h.hexdigest()==v['sha256'];assert complete['files'][n]=={'size':v['size'],'sha256':v['sha256']};actual[n]={'size':a.st_size,'sha256':h.hexdigest()}
assert complete['revision']==j['revision'] and complete['source_receipt_sha256']==j['source_receipt_sha256'];fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
remaining=[q.name for q in root.glob('*.part')];assert not remaining
out={'state':'INDEPENDENT_A800_FRESH_ALL_13FILES_SHA_VERIFIED','started_epoch':start,'finished_epoch':time.time(),'host':os.uname().nodename,'target':str(root),'route':j['route'],'verified_files':len(actual),'verified_bytes':sum(v['size']for v in actual.values()),'files':actual,'source_receipt_sha256':j['source_receipt_sha256'],'repo':j['repo'],'revision':j['revision'],'prefix':j['prefix'],'complete_sha256':hashlib.sha256((root/'asset-migration-complete.json').read_bytes()).hexdigest(),'binding_sha256':hashlib.sha256((root/'asset-binding.json').read_bytes()).hexdigest(),'process_after_verification':subprocess.run(['ps','-p',str(s['pid']),'-o','pid,stat,args'],capture_output=True,text=True,timeout=5).stdout,'model_bytes_through_Mac':0,'gpu_jobs_submitted':0};print(json.dumps(out))'''
ssh_args=control.route('baai-a800','python3 -B -c '+shlex.quote(code));ssh_args[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6']
r=subprocess.run(ssh_args,input=json.dumps(expected),capture_output=True,text=True,timeout=180)
if r.returncode:
 print(json.dumps({'verification_exit':r.returncode,'stderr_tail':r.stderr[-1600:]}))
 raise RuntimeError('Independent target verification not complete; no model or credential retained')
lines=r.stdout.strip().splitlines();x=json.loads(lines[-1]);x['progress_samples']=[json.loads(line)for line in lines[:-1]];(D/'easy-a800-asset-final-independent.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items()if k!='files'}))
