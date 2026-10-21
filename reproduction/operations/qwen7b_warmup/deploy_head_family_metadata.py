"""Finite metadata-only phase freeze/input resolver; never submits GPU work."""
import argparse,base64,gzip,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
P='reproduction/operations/qwen7b_warmup/'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozenE1-head-objective-family-v1'
COUNT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-frozenE1-fit30016-hold5120-v2'
BRANCH='codex/frozenE1-head-family-20261021'

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('freeze','resolve','status'),required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 name=P+('direct_head_family_protocol.py'if a.action in ('freeze','status')else 'direct_head_family_inputs.py');raw=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo);assert raw==(repo/name).read_bytes()
 if a.action!='status':assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]==a.commit
 spec=importlib.util.spec_from_file_location('head_family_private_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 payload=base64.b64encode(gzip.compress(raw,mtime=0)).decode();values=dict(base=BASE,count=COUNT,commit=a.commit,name=Path(name).name,source_sha=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw),payload=payload,action=a.action)
 script=r'''
import pathlib,base64,gzip,hashlib,json,sys,runpy,fcntl,os
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);lock=(b/'metadata.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
if v['action']=='status':
 out=dict(protocol=proof(b/'family-protocol.json')if(b/'family-protocol.json').exists()else None,inputs=proof(b/'coarse-fork-inputs.json')if(b/'coarse-fork-inputs.json').exists()else None)
 print(json.dumps(out));raise SystemExit(0)
code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True);raw=gzip.decompress(base64.b64decode(v['payload']));assert len(raw)==v['source_bytes']and hashlib.sha256(raw).hexdigest()==v['source_sha'];p=code/v['name']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw);p.chmod(0o444)
pub=dict(commit=v['commit'],source=proof(p),GPU_submitted=False);q=code/'publication.json';encoded=json.dumps(pub,sort_keys=True).encode()+b'\n'
if q.exists():assert q.read_bytes()==encoded
else:q.write_bytes(encoded)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
sys.argv=[str(p),'--base',str(b),'--baseline',v['count'],'--code-commit',v['commit']]
runpy.run_path(str(p),run_name='__main__')
'''
 command='python3 -c '+shlex.quote('v='+repr(values)+'\n'+script);assert len(command.encode())<16000
 result=ctl.run(command,account='mou176',source_address=ip,timeout=55);rows=[json.loads(x)for x in result.stdout.decode(errors='replace').splitlines()if x.startswith('{')]
 if result.returncode or not rows:raise RuntimeError(result.stdout.decode(errors='replace')[-2200:]+result.stderr.decode(errors='replace')[-1800:])
 out=rows[-1];dest=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('head-family-'+a.action+'.json');dest.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
