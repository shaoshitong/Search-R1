"""Launch one explicitly authorized Soft-to-Hard pipeline; never duplicate."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

JOB = Path('outputs/grounding-eval-20261019/a800-soft-to-hard-full-v4.job.json')
CODE = r'''import pathlib,json,subprocess,os,time,sys,hashlib,fcntl,struct,urllib.request
payload=json.load(sys.stdin);job=payload['job'];b=pathlib.Path('/data/shaoshitong/grounding-eval-20261019');r=pathlib.Path(job['runtime']);out=pathlib.Path(job['pipeline_output'])
assert job['source_commit']=='21b9ccf4939803c41ed3351415cd90616709735c'
assert job['run_id']=='1db662001a0f4d998928bcf6f5f33c9d'and job['method']=='Soft-to-Hard'
assert job['normal_root']==str(b/'soft-to-hard-normal')and job['cf_output']==str(b/'soft-to-hard-cf-full-v4')
assert json.loads((b/'search-g1-pipeline-v4/complete.json').read_text())['exit_code']==0
receipt=json.loads((r/'deployment-receipt.json').read_text());assert receipt['source_commit']==job['source_commit']
for rel,expected in receipt['files'].items():assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==expected['sha256']
assert not pathlib.Path('/proc/930253').exists(),'Original G1 pipeline is still present'
assert not pathlib.Path('/proc/956071').exists(),'Original G1 CF worker is still present'
retriever=b/'retriever-owned-51bcf912';service=json.loads((retriever/'launch.json').read_text());assert service['pid']==941984
p=pathlib.Path('/proc/941984');s=(p/'stat').read_text();assert int(s[s.rfind(')')+2:].split()[19])==service['start_ticks']
assert (p/'cmdline').read_bytes().decode().split('\0')[:-1]==service['command']
parity=json.loads((retriever/'parity.json').read_text());assert parity['count']==16 and parity['all_equal']
with urllib.request.urlopen('http://127.0.0.1:19000/health',timeout=3)as f:health=json.loads(f.read())
assert health['ok']and health['pid']==941984 and health['run_id']=='grounding-f531f78d53c040f7ab6f7f3d6c417313'
assert health['gpu_shards']==2 and health['topk']==3 and health['index_dtype']=='float16'and health['encoder_dtype']=='float16'
if(out/'launch.json').exists():print(json.dumps(dict(status='ALREADY_LAUNCHED_INSPECT_ONLY',launch=json.loads((out/'launch.json').read_text()))));raise SystemExit(0)
for d in (out,pathlib.Path(job['normal_root']),pathlib.Path(job['cf_output'])):
 assert not d.exists()or not any(d.iterdir()),'Existing nonempty task output requires explicit inspection'
# Probe the exact shared UUID namespace before launch. The normal worker then
# independently reacquires this lease and rechecks fresh NVML before allocation.
fd=os.open('/data/shaoshitong/.search-g1-gpu-uuid-locks/'+job['gpu_uuid']+'.lock',os.O_RDWR|os.O_NOFOLLOW)
fcntl.fcntl(fd,fcntl.F_OFD_SETLK,struct.pack('hhqqi4x',fcntl.F_WRLCK,os.SEEK_SET,0,0,0))
gpu=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert gpu.strip()=='4, '+job['gpu_uuid']+', 4, 0',gpu
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
assert not any(x.split(',')[0].strip()==job['gpu_uuid']for x in compute.splitlines())
os.close(fd);out.mkdir(mode=0o700,exist_ok=True)
path=b/'soft-to-hard-full-v4.job.json';raw=json.dumps(job,sort_keys=True,indent=2).encode()+b'\n'
with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
cmd=[job['python'],'-m','reproduction.grounding_eval.pipeline','--job',str(path)]
env=dict(os.environ,PYTHONPATH=str(r),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8')
with(out/'driver.log').open('xb')as f:q=subprocess.Popen(cmd,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
s=(pathlib.Path('/proc')/str(q.pid)/'stat').read_text();ticks=int(s[s.rfind(')')+2:].split()[19])
v=dict(pid=q.pid,start_ticks=ticks,command=cmd,source_commit=job['source_commit'],deployment_receipt_sha256=hashlib.sha256((r/'deployment-receipt.json').read_bytes()).hexdigest(),job_sha256=hashlib.sha256(raw).hexdigest(),job_path=str(path),at=time.time(),gpu_inventory_before=gpu,retriever_health_before=health,owned_retriever_preserved=True,first_g1_shared_acceptance=job['shared_prerequisite_verified_by_parent'])
with(out/'launch.json').open('x')as f:json.dump(v,f);f.flush();os.fsync(f.fileno())
print(json.dumps(v))
'''


def main():
    spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    control = importlib.util.module_from_spec(spec); spec.loader.exec_module(control)
    r = subprocess.run(control.route('baai-a800', '/usr/bin/python3 -c ' + shlex.quote(CODE)),
        input=json.dumps({'job': json.loads(JOB.read_text())}), capture_output=True, text=True, timeout=25)
    value = dict(transport_exit=r.returncode, stderr=r.stderr,
                 launch=json.loads(r.stdout) if not r.returncode else r.stdout)
    Path('outputs/grounding-eval-20261019/a800-soft-to-hard-full-v4-launch.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
