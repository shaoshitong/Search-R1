"""Read only our exact bounded audit process identities, no other process actions."""
import os,pathlib,json,time
rows=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  if p.stat().st_uid!=os.getuid():continue
  raw=(p/'cmdline').read_bytes()
  if b'Independently bind viewed dev/fixedB labels'not in raw or int(p.name)==os.getpid():continue
  rows.append(dict(pid=int(p.name),start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19],command_prefix=raw[:160].decode(errors='replace')))
 except(OSError,ValueError):pass
print(json.dumps(dict(time=time.time(),own_audit_processes=rows,processes_changed=False,final4096_read=False)))
