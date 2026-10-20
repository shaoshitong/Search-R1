"""Stage/run only the fixed own extra-model controlled retirement helper."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
from . import shared_vram_control as old

NAME = 'operational_reads/source5_model_tp4_controlled_stop.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'execute', 'status'))
    parser.add_argument('--published-commit')
    parser.add_argument('--published-ref')
    args = parser.parse_args()
    source = Path(__file__).with_name('operational_reads') / Path(NAME).name
    raw = source.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
    publication = None
    if args.action != 'status':
        publication = old.verify_publication(args.published_commit, args.published_ref,
            {NAME: dict(bytes=len(raw), sha256=sha)})
    destination = old.BASE + '/preparation/source5-model-tp4-to-tp1-v1/control/' + sha + '/' + source.name
    inner = 'BASE=' + repr(old.BASE) + '\nDEST=' + repr(destination) + '\nSHA=' + repr(sha) + '\nACTION=' + repr(args.action) + '\nDATA=' + repr(base64.b64encode(raw).decode()) + '\nPUBLICATION=' + repr(publication) + '\n' + r'''
import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
path=Path(DEST)
if ACTION=='stage':
 data=base64.b64decode(DATA);assert hashlib.sha256(data).hexdigest()==SHA
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  with path.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
 assert path.is_file()and not path.is_symlink()and hashlib.sha256(path.read_bytes()).hexdigest()==SHA
 os.chmod(path,0o555)
 print(json.dumps(dict(status='CONTROLLED_STOP_HELPER_CODE_FULL_SHA_PASS',path=str(path),bytes=len(data),sha256=SHA,publication=PUBLICATION)))
elif ACTION=='execute':
 assert path.is_file()and not path.is_symlink()and hashlib.sha256(path.read_bytes()).hexdigest()==SHA
 result=subprocess.run(['/usr/bin/python3',str(path)],text=True,capture_output=True,timeout=100)
 print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr);raise SystemExit(result.returncode)
else:
 root=Path(BASE)/'preparation/source5-model-tp4-to-tp1-v1'
 rows={}
 for name in('intent.json','receipt.json','complete.json'):
  p=root/name
  if p.exists():
   data=p.read_bytes();rows[name]=dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),value=json.loads(data))
 print(json.dumps(dict(status='READ_ONLY_FIXED_CONTROLLED_STOP_STATUS',records=rows)))
'''
    wrapper = ('import subprocess,sys\nr=subprocess.run(' + repr(['nsenter','--target','1490214','--mount','--net','--','/usr/bin/python3','-'])
        + ',input=' + repr(inner) + ',text=True,capture_output=True,timeout=105)\nprint(r.stdout,end="");print(r.stderr,end="",file=sys.stderr);raise SystemExit(r.returncode)\n')
    output = old.ROOT/'work'/('qwen7b-source5-tp4-retire-%s-%d.json' % (args.action,time.time_ns()))
    try:
        result = old.transport(wrapper,'h100-5')
    except subprocess.TimeoutExpired as error:
        output.write_text(json.dumps(dict(status='CONTROL_OUTCOME_UNKNOWN_QUERY_SAME_INTENT_NO_REPLAY',error=str(error),epoch=time.time()))+'\n')
        raise
    value = dict(action=args.action,source_sha256=sha,publication=publication,returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,epoch=time.time(),receipt=str(output))
    output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
    if result.returncode:raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
