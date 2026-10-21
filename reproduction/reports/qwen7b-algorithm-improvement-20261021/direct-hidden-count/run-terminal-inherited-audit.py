"""Stage published stdlib-only acceptance; never load tensor on login."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,zlib
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--commit',required=True);args=p.parse_args()
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2'
files={}
for name in ('audit_direct_hidden_count.py','audit_direct_hidden_terminal.py'):
 data=subprocess.check_output(['git','show',args.commit+':reproduction/operations/qwen7b_warmup/'+name],cwd=ROOT)
 files[name]=dict(payload=base64.b64encode(zlib.compress(data)).decode(),sha256=hashlib.sha256(data).hexdigest())
values=dict(base=BASE,commit=args.commit,files=files)
body='''import pathlib,json,base64,zlib,hashlib,fcntl,importlib.util,sys
b=pathlib.Path(values['base']);d=b.parent/('terminal-inherited-audit-'+values['commit'][:12]);d.mkdir(exist_ok=True)
lock=(d/'operator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pub=json.loads((b/'source-publication.json').read_text());assert pub['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e'
sys.path.insert(0,pub['code_root'])
for name,ref in values['files'].items():
 data=zlib.decompress(base64.b64decode(ref['payload']));assert hashlib.sha256(data).hexdigest()==ref['sha256'];p=d/name
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data);p.chmod(0o444)
 module_name='reproduction.operations.qwen7b_warmup.'+name[:-3]
 spec=importlib.util.spec_from_file_location(module_name,str(p));m=importlib.util.module_from_spec(spec);sys.modules[module_name]=m;spec.loader.exec_module(m)
module=sys.modules['reproduction.operations.qwen7b_warmup.audit_direct_hidden_terminal']
receipt=b/'evaluations/cursor-000938/terminal-inherited-independent-audit.json'
if receipt.exists():raise RuntimeError('Existing immutable terminal acceptance: inspect it before any repeat')
result=module.audit(b,'743872')
assert 'torch' not in sys.modules
print(json.dumps(dict(status=result['status'],mode=result['audit_mode'],host=result['host'],checkpoint=result['checkpoint'],metrics=result['metrics'],audit=module.a.proof(receipt),source_commit=values['commit'],code_files={n:{'sha256':r['sha256']} for n,r in values['files'].items()})))
'''
command='PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -c '+shlex.quote('values='+repr(values)+'\n'+body)
if len(command.encode())>=16000:raise ValueError('Code-only control exceeds16KB')
spec=importlib.util.spec_from_file_location('terminal_inherited_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();r=ctl.run(command,account='mou176',source_address=ip,timeout=90)
if r.returncode:raise RuntimeError(r.stderr[-2200:])
value=json.loads(r.stdout.splitlines()[-1]);(OUT/'actual-cursor-000938-terminal-inherited-audit.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
