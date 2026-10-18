"""Publish only immutable run-owned read-only audit operators; never edit runtime."""
import argparse,base64,hashlib,json,shlex,subprocess,time
from pathlib import Path
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--final-only',action='store_true');args=parser.parse_args()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/prefix20-40-80-cp50-20261018'],text=True).split()[0];assert commit==remote
names=['hpc3-two-stage-five-acceptance-20261018.py','hpc3-two-stage-milestone-readback-20261018.py','hpc3-two-stage-milestone-qa-acceptance-20261018.py','hpc3-two-stage-periodic-probe-readback-20261018.py']
if args.final_only:names=['hpc3-two-stage-final51713-acceptance-20261018.py']
scope='final' if args.final_only else 'milestone-periodic'
items=[]
for name in names:
 raw=(Path('docs')/name).read_bytes();h=hashlib.sha256(raw).hexdigest();items.append(dict(name=name,sha256=h,bytes=len(raw),data=base64.b64encode(raw).decode()))
code=r'''
import base64,hashlib,json,os,time
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4');assert b.resolve()==b and b.stat().st_uid==os.getuid()
p=b/'preparation/operators';p.mkdir(exist_ok=True);assert p.resolve()==p and p.stat().st_uid==os.getuid();result=[]
for item in ITEMS:
 raw=base64.b64decode(item['data']);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];path=p/(Path(item['name']).stem+'-'+item['sha256'][:12]+'.py')
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];result.append({k:item[k] for k in ['name','sha256','bytes']}|dict(path=str(path)))
out=dict(status='INDEPENDENT_AUDIT_OPERATORS_STAGED_NOT_EXECUTED',operator_source_commit=COMMIT,runtime_source_commit='2000d46a530c64d78be7802bf11175a68b65effe',files=result,checked_unix=time.time(),training_code_modified=False)
manifest=p/('audit-operators-'+COMMIT[:12]+'-'+SCOPE+'.json')
if not manifest.exists():
 with manifest.open('x') as f:json.dump(out,f,sort_keys=True)
else:
 old=json.loads(manifest.read_text());assert old['files']==result;out=old
print(json.dumps(out))
'''.replace('ITEMS',repr(items)).replace('COMMIT',repr(commit)).replace('SCOPE',repr(scope))
a=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip();r=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=a,timeout=25)
if r.returncode:raise RuntimeError(r.stdout.decode()+r.stderr.decode())
v=json.loads(r.stdout);registry=Path(OUTPUT)/'two-stage-r4-independent-audit-operators.json'
old=json.loads(registry.read_text()) if registry.exists() else {};merged={item['name']:item for item in old.get('files',[])}
merged.update({item['name']:item for item in v['files']});registry.write_text(json.dumps(dict(v,files=list(merged.values())),indent=2)+'\n');print(json.dumps(v))
