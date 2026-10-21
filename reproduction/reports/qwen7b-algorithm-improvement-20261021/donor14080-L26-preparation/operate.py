"""Immutable CPU prepare/stage/status only; deliberately no GPU submit action."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,zlib
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent
CONTROL=R.parents[1];TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/donor14080-independent-L26-prepared-v1';BRANCH='codex/donor14080-l26-prepare-20261021'
NAMES=['prepare_donor14080_l26.py','capture_donor14080_l26.py','test_prepare_donor14080_l26.py']
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','status'),required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0]
 if remote!=a.commit and subprocess.run(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R).returncode:raise ValueError('Exact published source required')
 s=importlib.util.spec_from_file_location('donor14080_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(script):
  command='python3 -c '+shlex.quote(script)
  if len(command.encode())>=16000:raise ValueError('Bounded control command exceeded')
  result=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-1200:]+result.stdout.decode(errors='replace')[-1200:])
  return json.loads(result.stdout.decode().splitlines()[-1])
 values=dict(base=TARGET,commit=a.commit,action=a.action)
 if a.action=='stage':
  code={n:subprocess.check_output(['git','show',a.commit+':reproduction/operations/qwen7b_warmup/'+n],cwd=R).decode()for n in NAMES}
  for n,v in code.items():assert(R/'reproduction/operations/qwen7b_warmup'/n).read_text()==v
  wire=zlib.compress(json.dumps(code).encode());digest=hashlib.sha256(wire).hexdigest()
  for offset in range(0,len(wire),6000):
   own=dict(values,offset=offset,part=base64.b64encode(wire[offset:offset+6000]).decode(),total=len(wire),wire_sha=digest)
   call('v='+repr(own)+'\n'+r'''import pathlib,json,hashlib,base64,os
b=pathlib.Path(v['base']);b.mkdir(parents=True,exist_ok=True);p=b/('code-'+v['commit'][:12]+'.upload');raw=base64.b64decode(v['part']);size=p.stat().st_size if p.exists()else 0
if size==v['offset']:
 with p.open('ab')as f:f.write(raw);f.flush();os.fsync(f.fileno())
else:
 assert size>=v['offset']+len(raw)
 with p.open('rb')as f:f.seek(v['offset']);assert f.read(len(raw))==raw
print(json.dumps(dict(status='SOURCE_CHUNK_ACCEPTED',offset=v['offset'])))''')
  values.update(wire_sha=digest,wire_bytes=len(wire))
 worker=r'''import pathlib,json,hashlib,fcntl,subprocess,sys,os,time,zlib
b=pathlib.Path(v['base']);code=b/('code-'+v['commit'][:12])
def fp(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(p.read_bytes())
if v['action']=='stage':
 raw=(b/('code-'+v['commit'][:12]+'.upload')).read_bytes();assert len(raw)==v['wire_bytes']and hashlib.sha256(raw).hexdigest()==v['wire_sha'];files=json.loads(zlib.decompress(raw));code.mkdir(exist_ok=True);refs={}
 for name,text in files.items():
  compile(text,name,'exec');p=code/name
  if p.exists():assert p.read_text()==text
  else:p.write_text(text);p.chmod(0o444)
  refs[name]=fp(p)
 out=dict(status='DONOR14080_IMMUTABLE_CPU_PREPARE_AND_FUTURE_CAPTURE_CODE_STAGED',commit=v['commit'],code_root=str(code),files=refs,GPU_submit_supported=False);p=code/'source-publication.json'
 if p.exists():assert read(p)==out
 else:p.write_text(json.dumps(out,sort_keys=True)+'\n')
 print(json.dumps(out));raise SystemExit(0)
pub=read(code/'source-publication.json');assert pub['commit']==v['commit']
for item in pub['files'].values():assert fp(pathlib.Path(item['path']))==item
if v['action']=='prepare':
 with(b/'launch.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(b/'launch-intent.json').exists():print(json.dumps(dict(status='EXISTING_CPU_PREPARATION_INTENT_PRESERVED',intent=read(b/'launch-intent.json'))));raise SystemExit(0)
  intent=dict(source_publication=fp(code/'source-publication.json'),epoch=time.time());(b/'launch-intent.json').write_text(json.dumps(intent))
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  log=(b/'prepare.log').open('ab',buffering=0);child=subprocess.Popen([sys.executable,'-u',str(code/'prepare_donor14080_l26.py'),'--prepare-only'],cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env=env);st=pathlib.Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split();out=dict(status='DONOR14080_CPU_PROJECTION_STARTED_NOT_YET_READY',pid=child.pid,birth_ticks=st[19],code_root=str(code),source_publication=intent['source_publication'],epoch=time.time());(b/'launch.json').write_text(json.dumps(out));print(json.dumps(out));raise SystemExit(0)
out=dict(status='ACTUAL_READ_ONLY_DONOR14080_CPU_PREPARATION_STATUS',base=str(b),epoch=time.time())
for name in('launch.json','progress.json','preparation.json','physical-source-final-readback.json'):
 if(b/name).exists():out[name]=dict(ref=fp(b/name),value=read(b/name))
launch=out.get('launch.json',{}).get('value',{});proc=pathlib.Path('/proc/%s'%launch.get('pid',0));out['same_process_birth']=False
if(proc/'stat').exists():out['same_process_birth']=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==launch['birth_ticks']
if(b/'prepare.log').exists():out['log_tail']=(b/'prepare.log').read_text()[-1200:]
print(json.dumps(out))'''
 value=call('v='+repr(values)+'\n'+worker);(HERE/('actual-'+a.action+'.json')).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
