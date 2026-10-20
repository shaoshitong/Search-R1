"""Deploy the published small pool-freezing tool; all question data stay HPC3."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]==commit
 rel='reproduction/operations/qwen7b_warmup/freeze_confirmation_pools.py';raw=(repo/rel).read_bytes();assert raw==subprocess.check_output(['git','show',commit+':'+rel],cwd=repo)
 spec=importlib.util.spec_from_file_location('control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 code='commit='+repr(commit)+'\nraw64='+repr(base64.b64encode(raw).decode())+'\nsha256='+repr(hashlib.sha256(raw).hexdigest())+'\n'+REMOTE
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();r=m.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=45)
 result=[json.loads(x) for x in r.stdout.decode().splitlines() if x.startswith('{')]
 if r.returncode or len(result)!=1:raise RuntimeError(r.stderr.decode()[-1000:]+' '+r.stdout.decode()[-1000:])
 out=a.control_root/'outputs/qwen7b-warmup-20261019/confirmation-pool-freeze.json';out.write_text(json.dumps(result[0],indent=2)+'\n');print(json.dumps(result[0]))

REMOTE=r'''
import pathlib,hashlib,base64,subprocess,json,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');b=B/'algorithm-improvement-20261021';root=B/'control-tools'/('confirmation-freeze-'+commit[:12]);root.mkdir(parents=True,exist_ok=True)
raw=base64.b64decode(raw64);assert hashlib.sha256(raw).hexdigest()==sha256;p=root/'freeze_confirmation_pools.py'
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
r=subprocess.run(['python3',str(p),'--training-base',str(B),'--base',str(b),'--source-commit',commit],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35)
assert r.returncode==0,r.stdout+r.stderr
result=json.loads(r.stdout);result['deployed_source']=dict(path=str(p),sha256=sha256,commit=commit)
# Re-read each sealed artifact, checking SHA and pairwise/source-ID separation.
manifest=json.loads((b/'confirmation/manifest.json').read_text());sets=[]
for name,item in manifest['pools'].items():
 path=pathlib.Path(item['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];rows=json.loads(path.read_text())['questions'];ids={x['source_question_id'] for x in rows};assert len(ids)==2048
 assert all(not ids&s for s in sets);sets.append(ids)
result['second_read_SHA_and_disjointness_passed']=True;result['verified_at']=time.time();print(json.dumps(result))
'''
if __name__=='__main__':main()
