"""Execute the human-authorized exact invalid-run output cleanup, not asset deletion."""
import argparse
import json
from pathlib import Path
import time
from reproduction.h1008_qprobe.management_transport import remote

SCRIPT = r'''
import hashlib,json,os,pathlib,shutil,subprocess,time
p=PAYLOAD;plan=p['plan'];rid=plan['run_id'];root=pathlib.Path('/mnt/data/shaoshitong')
def read(q):return json.loads(q.read_text())
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def call(args):return subprocess.run(args,capture_output=True,text=True,timeout=30)
assert rid=='f3ee1435e5014ba1969387c91878893a' and len(plan['bases'])==7
assert p['authorization']['message_id']=='01a14849-d31c-7a80-b223-6a46bc3a5a08'
n=call(['scontrol','show','job','3427','-o']);assert n.returncode==0 and 'JobState=CANCELLED' in n.stdout
q=call(['squeue','-h','-o','%i|%j|%N']);assert q.returncode==0 and 'g1-prefix200-f3ee1435e501'not in q.stdout
protected={}
for name in plan['protected_external_assets']:
 path=pathlib.Path(name);info=path.stat();protected[name]=dict(inode=info.st_ino,bytes=info.st_size,mtime_ns=info.st_mtime_ns)
targets=[]
for name in plan['bases']:
 b=pathlib.Path(name);assert b.parent==root and b.resolve()==b and b.name.startswith('search-g1-prefix-diversity200-20261016-f3ee1435e501')
 config=b/'preparation/launch-config.json';assert sha(config)==plan['config_sha256'][name]and read(config)['run_id']==rid
 for child in plan['delete_child_names']:
  assert child in ('actor','probe','outers','logs','recovered-validation','retrieval-cache','cache','tmp','grounding','final-evaluation','validation-retries')
  path=b/child
  if not path.exists():continue
  assert not path.is_symlink()and path.resolve()==path
  assert not any(path==pathlib.Path(x)or path in pathlib.Path(x).parents for x in protected)
  size=call(['du','-sk',str(path)]);assert size.returncode==0
  targets.append(dict(path=str(path),allocated_kib=int(size.stdout.split()[0]),directory=path.is_dir()))
 # Run-owned telemetry and audit .log files are invalid-run logs too.
 for path in b.glob('acceptance*/*.log'):
  assert not path.is_symlink();targets.append(dict(path=str(path),allocated_kib=(path.stat().st_size+1023)//1024,directory=False))
 for rel in ('preparation/resource5s-v1/samples.jsonl','preparation/resource5s-v1/unit.log'):
  path=b/rel
  if path.exists():assert not path.is_symlink();targets.append(dict(path=str(path),allocated_kib=(path.stat().st_size+1023)//1024,directory=False))
intent=root/'search-g1-invalid-f3ee-i1i6-delete-intent-20261017.json'
assert not intent.exists(),'Existing cleanup intent requires reconciliation, never blind replay'
before=dict(created_unix=time.time(),authorization=p['authorization'],plan=plan,targets=targets,protected=protected,native=n.stdout)
with intent.open('x')as f:json.dump(before,f,sort_keys=True)
deleted=[]
for row in targets:
 path=pathlib.Path(row['path'])
 if row['directory']:shutil.rmtree(path)
 else:path.unlink()
 deleted.append(row)
for name,info in protected.items():
 st=pathlib.Path(name).stat();assert dict(inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns)==info
assert all(not pathlib.Path(row['path']).exists()for row in deleted)
result=dict(status='EXACT_INVALID_RUN_OUTPUTS_DELETED_SHARED_ASSETS_RETAINED',run_id=rid,deleted=deleted,
 deleted_allocated_kib_sum_before=sum(x['allocated_kib']for x in deleted),
 byte_scope='du allocated sum is not a unique physical-reclaimed count because hardlinks exist',
 protected_after_stat_verified=protected,kept_code_and_small_preparation_audits=True,completed_unix=time.time())
receipt=root/'search-g1-invalid-f3ee-i1i6-delete-result-20261017.json'
with receipt.open('x')as f:json.dump(result,f,sort_keys=True)
print(json.dumps(result))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--authorization',required=True)
    p.add_argument('--receipts',required=True);p.add_argument('--control-path',required=True);a=p.parse_args()
    payload=dict(plan=json.loads(Path(a.plan).read_text()),authorization=json.loads(Path(a.authorization).read_text()))
    dest=Path(a.receipts);dest.mkdir(parents=True,exist_ok=True);t=dest/('delete-'+str(time.time_ns())+'-transport.json')
    value=json.loads(remote(SCRIPT.replace('PAYLOAD',repr(payload)),t,control_path=a.control_path,target='h100-10'))
    result=t.with_name(t.name.replace('-transport','-result'));result.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(evidence=str(result),status=value['status'],deleted_paths=len(value['deleted']),
        protected_assets=len(value['protected_after_stat_verified']),allocated_kib_sum_before=value['deleted_allocated_kib_sum_before'])))

if __name__=='__main__':main()
