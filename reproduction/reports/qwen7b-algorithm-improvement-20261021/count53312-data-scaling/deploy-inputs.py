"""Stage published metadata preparation only; no Slurm or model operation."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,subprocess,sys
Q='reproduction/reports/qwen7b-algorithm-improvement-20261021/count53312-data-scaling/'
p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[4];O=a.control_root/'outputs/qwen7b-warmup-20261019';branch='codex/count53312-data-scaling-20261021'
assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]==a.commit
payload={}
for n in ('PROTOCOL.json','SOURCE-INPUTS.json','prepare-inputs.py','deploy-inputs.py','PROTOCOL.md'):
 data=subprocess.check_output(['git','show',a.commit+':'+Q+n],cwd=repo);assert data==(repo/Q/n).read_bytes();payload[n]=base64.b64encode(data).decode()
blob=gzip.compress(json.dumps(payload,sort_keys=True).encode(),mtime=0);root='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/count53312-data-scaling-inputs-v1'
def call(script):
 path=O/'scale-inputs-current-control.py';path.write_text(script);r=subprocess.run(['python3',str(O/'balanced-control.py'),str(path),'--timeout','50'],capture_output=True,text=True,timeout=65)
 if r.returncode:raise RuntimeError(r.stdout[-1800:]+r.stderr[-1800:])
 lines=[json.loads(x)for x in r.stdout.splitlines()if x.startswith('{')];assert lines,r.stdout[-1800:];return lines[-1]
for i,start in enumerate(range(0,len(blob),4500)):
 part=blob[start:start+4500];script="import pathlib,base64,hashlib,json\nb=pathlib.Path("+repr(root)+");b.mkdir(exist_ok=True);code=b/"+repr('code-'+a.commit[:12])+";code.mkdir(exist_ok=True);v=base64.b64decode("+repr(base64.b64encode(part).decode())+");p=code/"+repr('part-%04d'%i)+"\nif p.exists():assert p.read_bytes()==v\nelse:p.write_bytes(v)\nprint(json.dumps(dict(part="+str(i)+",bytes=len(v))))\n";call(script)
script="import pathlib,json,hashlib,base64,gzip,subprocess,os,fcntl\nb=pathlib.Path("+repr(root)+");code=b/"+repr('code-'+a.commit[:12])+";lock=(b/'operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\nv=b''.join((code/('part-%04d'%i)).read_bytes()for i in range("+str((len(blob)+4499)//4500)+"));assert hashlib.sha256(v).hexdigest()=="+repr(hashlib.sha256(blob).hexdigest())+"\nfiles=json.loads(gzip.decompress(v));refs={}\nfor n,s in files.items():\n raw=base64.b64decode(s);p=code/n\n if p.exists():assert p.read_bytes()==raw\n else:p.write_bytes(raw);p.chmod(0o444)\n refs[n]=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())\npub=dict(commit="+repr(a.commit)+",code=str(code),files=refs,GPU_submission=False);p=b/'source-publication.json';raw=json.dumps(pub,sort_keys=True,separators=(',',':')).encode()+bytes([10])\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nos.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);r=subprocess.run(['python3',str(code/'prepare-inputs.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=42);(b/'prepare.log').write_text(r.stdout+r.stderr)\nif r.returncode:raise RuntimeError(r.stderr[-2400:])\nx=json.loads(r.stdout);x['source_publication']=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());x['source_commit']="+repr(a.commit)+";print(json.dumps(x,sort_keys=True))\n"
x=call(script);out=repo/Q/'actual-READY.json';out.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps(x,sort_keys=True))
