import pathlib,json,sys,hashlib,re,time,os,subprocess,shlex
p=json.loads(sys.stdin.read());assert p['commit']==p['remote_head'] and re.fullmatch('[0-9a-f]{40}',p['commit'])
assert set(p['files'])=={'qualify_fa3_kernel_cuda.py','run_fa3_kernel_probe.sh'}
base=pathlib.Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
root=base/'diagnostics/fa3-kernel-20261021-r1';root.mkdir(parents=True,exist_ok=True)
assert root.resolve()==root
if (root/'submit-claim.json').exists():
 print(json.dumps({'status':'ALREADY_ATTEMPTED_INSPECT_EXISTING','root':str(root),'claim':json.loads((root/'submit-claim.json').read_text())}));sys.exit(0)
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
files={}
for name,row in p['files'].items():
 content=row['source'].encode();assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256'];path=root/name
 with path.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];files[name]={'path':str(path),'bytes':len(content),'sha256':row['sha256']}
interface='/data/user/jxu957/experiments/topd-20261008/runtime/env/lib/python3.10/site-packages/flash_attn_interface.py'
cudart='/share/anaconda3/envs/flash_attn3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12'
assert hashlib.sha256(pathlib.Path(interface).read_bytes()).hexdigest()=='a50f564fa1fc4cdd85ffa61f79e913a463c2066e519f9c5e8bc2e5c74457fa38'
assert pathlib.Path(cudart).is_file()
command=['sbatch','--parsable','--partition=emergency_acd','--qos=formal-user','--account=jxu957','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=8','--mem=64G','--time=00:20:00','--job-name=easy7b-fa3-kernel','--export=NONE','--chdir='+str(root),'--output='+str(root/'slurm-%j.log'),'--error='+str(root/'slurm-%j.log'),str(root/'run_fa3_kernel_probe.sh'),str(root),str(base/'env/bin/python'),interface,cudart]
claim={'source_commit':p['commit'],'remote_head':p['remote_head'],'files':files,'command':command,'root':str(root),'diagnostic_only':True,'training_modified':False,'model_loaded':False,'created_unix':time.time()}
write(root/'submit-claim.json',claim)
r=subprocess.run(command,capture_output=True,text=True,timeout=30);receipt=dict(claim,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time())
if r.returncode==0:
 job=r.stdout.strip().split(';')[0];assert job.isdecimal();receipt['job']=job
 with (root/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
write(root/'submit-receipt.json',receipt)
print(json.dumps(receipt))
