"""Fresh read of the identified589-byte terminal receipt, safe primitive fields."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
WORKER = '''import pathlib,json,hashlib,time,subprocess
B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
p=B/'pool/jobs/job-relay-slot-gpu4-tp1-3731224/attempt-1792555662309369517-3731224/complete.json'
raw=p.read_bytes();v=json.loads(raw)
assert len(raw)==589 and hashlib.sha256(raw).hexdigest()=='aedf9bc4b205b8c6afad71c859dabe8940ada680180bf4b8dde03817bdb08de6'
assert all(isinstance(x,(str,int,float,bool,type(None)))for x in v.values())
unit='sst-q7b-g5-relaygen-queued-owner-v2-downstream-r1-70188b8e50fd.service'
r=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','Result','-p','ExecMainStatus'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
assert r.returncode==0
owner=dict(line.split('=',1)for line in r.stdout.splitlines()if'='in line)
print(json.dumps(dict(status='FRESH_SUPPLEMENTARY_TERMINAL_RECEIPT_VERIFIED',epoch=time.time(),terminal_receipt=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),fields=v),owner_current=owner,producer_modified=False,payload_read=False,GPU_operation=False),sort_keys=True))
'''

def main():
    spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctl)
    result = subprocess.run(ctl.route('h100-10', 'python3 -c '+shlex.quote(WORKER)),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-700:])
    value = json.loads(result.stdout.decode().splitlines()[-1])
    (HERE/'actual-supp-final-small.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, sort_keys=True))

if __name__ == '__main__':
    main()
