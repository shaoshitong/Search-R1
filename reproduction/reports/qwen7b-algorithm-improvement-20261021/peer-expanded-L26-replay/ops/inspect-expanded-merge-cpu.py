"""Actual owned CPU merge phase/IO without opening model or tensor payload."""
from pathlib import Path
import json,os,time
p=Path('/proc/3143539');result=dict(pid=3143539,time=time.time())
if p.exists():
 assert p.stat().st_uid==os.getuid();result['status']=[line for line in (p/'status').read_text().splitlines() if line.startswith(('State:','VmRSS:','VmHWM:','Threads:'))];result['io']=(p/'io').read_text();result['command_contains_expected_module']='peer_expanded_l26_merge' in (p/'cmdline').read_bytes().decode(errors='replace');result['files']=[]
 for fd in (p/'fd').iterdir():
  try:
   name=os.readlink(str(fd))
   if 'peer-expanded-L26-replay-v1' in name or 'peer-newseed8192' in name:result['files'].append(name)
  except OSError:pass
else:result['process_exists']=False
print(json.dumps(result))
