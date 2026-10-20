"""Stage published independent aggregate auditor and execute only after real complete."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','audit'),required=True);p.add_argument('--control-root',type=Path,required=True);a=p.parse_args();C=a.control_root;R=C/'work/qwen7b-probe-warmup-20261019';branch='codex/peer8192-long-prefix-audit-20261021'
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=R,text=True).split()[0];assert remote==a.commit
spec=importlib.util.spec_from_file_location('long_prefix_audit_control',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();B='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-long-prefix-causal-diagnostic-v1';rel='reproduction/operations/qwen7b_warmup/audit_long_prefix_diagnostic.py';data=subprocess.check_output(['git','show',a.commit+':'+rel],cwd=R);want=hashlib.sha256(data).hexdigest();target=B+'/audit-code-'+a.commit[:12]+'.py'
def call(s):
 cmd='python3 -c '+shlex.quote(s);assert len(cmd.encode())<16000;r=control.run(cmd,account='mou176',source_address=address,timeout=55)
 if r.returncode:raise RuntimeError('Actual aggregate audit failed: '+r.stderr.decode(errors='replace')[-2000:]+' '+r.stdout.decode(errors='replace')[-2000:])
 rows=[json.loads(x) for x in r.stdout.decode().splitlines() if x.startswith('{')];assert rows;return rows[-1]
if a.action=='stage':
 for i,start in enumerate(range(0,len(data),4096)):
  x=data[start:start+4096];part=target+'.part-%04d'%i;s="import base64,hashlib,json\nfrom pathlib import Path\np=Path(%r);x=base64.b64decode(%r);assert hashlib.sha256(x).hexdigest()==%r\nif p.exists():assert p.read_bytes()==x\nelse:p.write_bytes(x)\nprint(json.dumps({'status':'INDEPENDENT_AUDIT_CODE_PART_STAGED'}))"%(part,base64.b64encode(x).decode(),hashlib.sha256(x).hexdigest());call(s)
 s="import hashlib,json\nfrom pathlib import Path\np=Path(%r);parts=sorted(p.parent.glob(p.name+'.part-*'));x=b''.join(v.read_bytes() for v in parts);assert hashlib.sha256(x).hexdigest()==%r\nif p.exists():assert p.read_bytes()==x\nelse:p.write_bytes(x);p.chmod(0o444)\nfor v in parts:v.unlink()\nprint(json.dumps(dict(status='PUBLISHED_INDEPENDENT_AUDITOR_SHA_VERIFIED',source_commit=%r,path=str(p),bytes=len(x),sha256=hashlib.sha256(x).hexdigest())))"%(target,want,a.commit);result=call(s)
else:
 s="import hashlib,subprocess,sys\nfrom pathlib import Path\np=Path(%r);assert hashlib.sha256(p.read_bytes()).hexdigest()==%r\nsubprocess.check_call([sys.executable,str(p)])"%(target,want);result=call(s)
out=C/'outputs/qwen7b-warmup-20261019/reports'/('long-prefix-independent-'+a.action+'-latest.json');out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
