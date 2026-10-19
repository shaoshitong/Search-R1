"""Prepared synthetic 8 MiB probe; requires --execute. No checkpoint bytes/files."""
import argparse,hashlib,importlib.util,json,shlex,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
 if not a.execute:raise SystemExit('Prepared only: add --execute for synthetic 8 MiB traffic (not weights)')
 spec=importlib.util.spec_from_file_location('grounding_controller','/Users/shaoshitong/Documents/remote_codex_control/control.py');controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
 n=8*1024*1024
 source="import os,sys,hashlib,json;d=os.urandom(%d);sys.stdout.buffer.write(d);sys.stdout.buffer.flush();sys.stderr.write('MEMORY_PROBE_JSON:'+json.dumps(dict(bytes=len(d),sha256=hashlib.sha256(d).hexdigest()))+chr(10))"%n
 sink="import sys,hashlib,json,time;t=time.monotonic();h=hashlib.sha256();n=0\nwhile True:\n d=sys.stdin.buffer.read(1024*1024)\n if not d:break\n n+=len(d);h.update(d)\nprint('MEMORY_PROBE_JSON:'+json.dumps(dict(bytes=n,sha256=h.hexdigest(),read_seconds=time.monotonic()-t)))"
 target=['/usr/bin/ssh','-F','/Users/shaoshitong/Documents/remote_codex_control/access/hpc3-20261008/ssh_config','-T','-S','/tmp/grounding-eval-20261019-501/hpc3-cm.sock','-oUser=mou176','-oBatchMode=yes','hpc3','python3 -B -c '+shlex.quote(sink)]
 t=time.monotonic();src=None;dst=None
 try:
  src=subprocess.Popen(controller.route('h100-10','python3 -B -c '+shlex.quote(source)),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  dst=subprocess.Popen(target,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  copied=0;first=None
  while True:
   block=src.stdout.read(1024*1024)
   if not block:break
   if first is None:first=time.monotonic()
   dst.stdin.write(block);copied+=len(block)
  dst.stdin.close();dst.stdin=None
  out,err=dst.communicate(timeout=30);src.stdout.close();sout,serr=src.communicate(timeout=30)
  def receipt(data):return json.loads(next(l.split('MEMORY_PROBE_JSON:',1)[1]for l in data.decode().splitlines()if'MEMORY_PROBE_JSON:'in l))
  s=receipt(serr);d=receipt(out)
  assert src.returncode==dst.returncode==0,(src.returncode,dst.returncode,err.decode()[-300:])
  assert copied==s['bytes']==d['bytes']==n and s['sha256']==d['sha256']
  elapsed=time.monotonic()-t
  result=dict(status='synthetic_8MiB_probe_verified',bytes=n,sha256=s['sha256'],elapsed_including_setup=elapsed,MBps_including_setup=n/elapsed/1e6,source='retained18/h100-10 via existing mgmt-host route',destination='HPC3 existing HKUST3 ControlMaster',mac_peak_bridge_chunk_bytes=1024*1024,weights_transferred=False,files_written_remotely=False,source_receipt=s,destination_receipt=d)
  Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 finally:
  for proc in (src,dst):
   if proc is not None and proc.poll()is None:
    proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
if __name__=='__main__':main()
