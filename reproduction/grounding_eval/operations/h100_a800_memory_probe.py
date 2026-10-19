"""One authorized8MiB synthetic memory-only route probe; never weights."""
import hashlib,importlib.util,json,os,selectors,shlex,subprocess,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('controller','/Users/shaoshitong/Documents/remote_codex_control/control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
size=8*1024*1024
source="import os,sys,hashlib,json;d=os.urandom(%d);sys.stdout.buffer.write(d);sys.stdout.buffer.flush();sys.stderr.write('MEMORY_PROBE_JSON:'+json.dumps(dict(bytes=len(d),sha256=hashlib.sha256(d).hexdigest()))+chr(10))"%size
sink="import sys,hashlib,json,time;t=time.monotonic();h=hashlib.sha256();n=0\nwhile True:\n d=sys.stdin.buffer.read(1024*1024)\n if not d:break\n n+=len(d);h.update(d)\nprint('MEMORY_PROBE_JSON:'+json.dumps(dict(bytes=n,sha256=h.hexdigest(),read_seconds=time.monotonic()-t)))"
output=Path('outputs/grounding-eval-20261019/h100-a800-memory-probe-result.json');progress=output.with_name('h100-a800-memory-probe-progress.json')
src=dst=None;started=time.monotonic();copied=0;first=None;last=None;maximum=0;last_status=0
try:
 src=subprocess.Popen(c.route('h100-10','python3 -B -c '+shlex.quote(source)),stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 dst=subprocess.Popen(c.route('baai-a800','python3 -B -c '+shlex.quote(sink)),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 os.set_blocking(src.stdout.fileno(),False);os.set_blocking(dst.stdin.fileno(),False)
 sel=selectors.DefaultSelector();sel.register(src.stdout,selectors.EVENT_READ,'source');buffer=bytearray();eof=False;registered=False
 while not eof or buffer:
  elapsed=time.monotonic()-started
  if elapsed>120:raise TimeoutError('120s overall route-probe deadline')
  if buffer and not registered:sel.register(dst.stdin,selectors.EVENT_WRITE,'destination');registered=True
  if not buffer and registered:sel.unregister(dst.stdin);registered=False
  for key,_ in sel.select(timeout=.5):
   if key.data=='source':
    if len(buffer)>1024*1024:continue
    data=os.read(src.stdout.fileno(),65536)
    if not data:sel.unregister(src.stdout);eof=True
    else:
     if first is None:first=time.monotonic()
     buffer.extend(data);maximum=max(maximum,len(buffer))
   elif key.data=='destination' and buffer:
    n=os.write(dst.stdin.fileno(),buffer);del buffer[:n];copied+=n;last=time.monotonic()
  if elapsed-last_status>5:
   progress.write_text(json.dumps(dict(elapsed=elapsed,bytes_copied_into_sink_stdin=copied,source_eof=eof,buffered_bytes=len(buffer),source_rc=src.poll(),destination_rc=dst.poll(),verified=False)))
   last_status=elapsed
 sel.close();dst.stdin.close();dst.stdin=None
 out,err=dst.communicate(timeout=max(1,120-(time.monotonic()-started)));sout,serr=src.communicate(timeout=10)
 def receipt(blob):return json.loads(next(line.split('MEMORY_PROBE_JSON:',1)[1] for line in blob.decode().splitlines() if 'MEMORY_PROBE_JSON:' in line))
 sr,dr=receipt(serr),receipt(out)
 assert src.returncode==dst.returncode==0,(src.returncode,dst.returncode,err.decode()[-500:])
 assert copied==sr['bytes']==dr['bytes']==size and sr['sha256']==dr['sha256']
 elapsed=time.monotonic()-started;transfer=last-first
 result=dict(status='synthetic_8MiB_probe_verified',bytes=size,sha256=sr['sha256'],elapsed_including_setup=elapsed,MBps_including_setup=size/elapsed/1e6,bridge_first_to_last_byte_seconds=transfer,MBps_bridge_phase=size/transfer/1e6,maximum_python_binary_buffer_bytes=maximum,weights_transferred=False,files_written_remotely=False,source='retained18 via existing mgmt-host route',destination='baai-a800 existing controller route',source_receipt=sr,destination_receipt=dr)
except Exception as exc:
 result=dict(status='probe_failed',error=type(exc).__name__+':'+str(exc),elapsed_including_setup=time.monotonic()-started,bytes_copied_into_sink_stdin=copied,verified_bytes=None,throughput_MBps=None,maximum_python_binary_buffer_bytes=maximum,weights_transferred=False)
finally:
 for proc in [src,dst]:
  if proc is not None and proc.poll() is None:
   proc.terminate()
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
