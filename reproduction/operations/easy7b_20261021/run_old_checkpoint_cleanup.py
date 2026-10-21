"""Pinned RAM-credential bridge for the authorized old-checkpoint cleanup only."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import time


def main():
 p=argparse.ArgumentParser();p.add_argument('user',choices=['jxu957','mou176']);p.add_argument('mode',choices=['inventory-start','inventory-status','apply-start','apply-status','verify']);p.add_argument('--envelope',type=Path);args=p.parse_args()
 cwd=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
 here=Path(__file__).resolve().parent
 spec=importlib.util.spec_from_file_location('bridge',cwd/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py');bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
 uid={'jxu957':14090,'mou176':21930}[args.user]
 control='/tmp/authorized-old-checkpoints-20261021-'+str(uid)
 payload=''
 if args.mode=='verify':
  code=(here/'verify_authorized_old_checkpoint_cleanup.py').read_text()
 elif args.mode=='apply-start':
  assert args.envelope and args.envelope.is_file()
  payload=args.envelope.read_text();envelope=json.loads(payload)
  source=(here/'apply_authorized_old_checkpoint_cleanup.py').read_text()
  code="""import os,json,subprocess,time,hashlib,sys
from pathlib import Path
assert os.getuid()==UID
root=Path(CONTROL);assert root.stat().st_uid==os.getuid()
source=SOURCE
payload=sys.stdin.read();envelope=json.loads(payload)
assert hashlib.sha256(envelope['plan'].encode()).hexdigest()==envelope['plan_sha256']
if (root/'apply-launch-claim.json').exists():
 print(json.dumps(dict(existing_claim=json.loads((root/'apply-launch-claim.json').read_text()),new_process_started=False)))
else:
 assert not(root/'apply-claim.json').exists()and not(root/'result.json').exists()
 (root/'apply-source.py').write_text(source);(root/'approved-plan-envelope.json').write_text(payload)
 inp=(root/'approved-plan-envelope.json').open('rb');out=(root/'apply.stdout').open('xb');err=(root/'apply.stderr').open('xb')
 child=subprocess.Popen(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B',str(root/'apply-source.py')],stdin=inp,stdout=out,stderr=err,start_new_session=True)
 fields=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
 claim=dict(pid=child.pid,start_ticks=int(fields[19]),uid=os.getuid(),started_unix=time.time(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),plan_sha256=envelope['plan_sha256'])
 with(root/'apply-launch-claim.json').open('x')as stream:json.dump(claim,stream);stream.flush();os.fsync(stream.fileno())
 print(json.dumps(dict(claim=claim,control=str(root),new_process_started=True)))
""".replace('UID',str(uid)).replace('CONTROL',repr(control)).replace('SOURCE',repr(source))
 elif args.mode=='apply-status':
  code="""import os,json,time
from pathlib import Path
assert os.getuid()==UID
root=Path(CONTROL);claim=json.loads((root/'apply-launch-claim.json').read_text());proc=Path('/proc',str(claim['pid']));same=False;state=None
if proc.exists():
 fields=(proc/'stat').read_text().rsplit(')',1)[1].split();state=fields[0];same=int(fields[19])==claim['start_ticks']and state!='Z'
result=dict(epoch=time.time(),control=str(root),claim=claim,same_birth=same,state=state,stderr=(root/'apply.stderr').read_text()[-2500:])
for name in ['result.json','failure.json','apply-claim.json']:
 if(root/name).is_file():result[name]=json.loads((root/name).read_text())
if(root/'unlinked.jsonl').is_file():
 rows=[json.loads(line)for line in(root/'unlinked.jsonl').read_text().splitlines()if line];result['unlinked_count']=len(rows);result['unlinked_last_path']=rows[-1]['path']if rows else None
print(json.dumps(result,ensure_ascii=False))
""".replace('UID',str(uid)).replace('CONTROL',repr(control))
 elif args.mode=='inventory-start':
  source=(here/'inventory_authorized_old_checkpoint_cleanup.py').read_text()
  code="""import os,json,subprocess,time,hashlib
from pathlib import Path
assert os.getuid()==UID
root=Path(CONTROL);root.mkdir(mode=0o700,exist_ok=True)
source=SOURCE
if (root/'inventory-claim.json').exists():
 print(json.dumps(dict(existing_claim=json.loads((root/'inventory-claim.json').read_text()),control=str(root),new_process_started=False)))
else:
 (root/'inventory-source.py').write_text(source)
 out=(root/'inventory.json').open('xb');err=(root/'inventory.stderr').open('xb')
 child=subprocess.Popen(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B',str(root/'inventory-source.py')],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 fields=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()
 claim=dict(pid=child.pid,start_ticks=int(fields[19]),uid=os.getuid(),started_unix=time.time(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),read_only=True)
 with (root/'inventory-claim.json').open('x')as stream:json.dump(claim,stream);stream.flush();os.fsync(stream.fileno())
 print(json.dumps(dict(claim=claim,control=str(root),new_process_started=True)))
""".replace('UID',str(uid)).replace('CONTROL',repr(control)).replace('SOURCE',repr(source))
 else:
  code="""import os,json,time
from pathlib import Path
assert os.getuid()==UID
root=Path(CONTROL);claim=json.loads((root/'inventory-claim.json').read_text());proc=Path('/proc',str(claim['pid']));same=False;state=None
if proc.exists():
 fields=(proc/'stat').read_text().rsplit(')',1)[1].split();state=fields[0];same=int(fields[19])==claim['start_ticks']and state!='Z'
result=dict(epoch=time.time(),control=str(root),claim=claim,same_birth=same,state=state,output_bytes=(root/'inventory.json').stat().st_size,stderr=(root/'inventory.stderr').read_text()[-2000:])
if not same:
 try:result['inventory']=json.loads((root/'inventory.json').read_text())
 except ValueError:result['incomplete_output']=True
print(json.dumps(result,ensure_ascii=False))
""".replace('UID',str(uid)).replace('CONTROL',repr(control))
 command=shlex.join(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code])
 r=bridge.run(args.user,command,payload,str(cwd/'work'),login_host='10.120.48.31',expected_host_identity=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid_namespace='pid:[4026531836]'),observation_timeout=55)
 target=cwd/'outputs/easy2560-qwen7b-20261021'/('old-cleanup-'+args.mode+'-'+args.user+'-'+str(int(time.time()))+'.json')
 target.write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 print(target.name,'returncode',r.returncode)
 if r.returncode:print(r.stderr[-1500:])
 else:
  d=json.loads(r.stdout)
  inv=d.pop('inventory',None)
  if inv:d['inventory_summary']=dict(groups=inv['groups'],files=len(inv['files']),texts=len(inv['preserved_text']),errors=len(inv['errors']),active_references=inv['active_references'])
  print(json.dumps(d,ensure_ascii=False))
 raise SystemExit(bool(r.returncode))

if __name__=='__main__':main()
