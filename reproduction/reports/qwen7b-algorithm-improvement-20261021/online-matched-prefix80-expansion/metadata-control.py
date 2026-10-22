"""Bounded read-only metadata control via existing approved SSH routes.

Unknown-outcome mutations must never use this automatic read retry adapter.
No training process, Mac proxy or network configuration is changed.
"""
import argparse,importlib.util,os,pathlib,shlex,signal,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('script',type=pathlib.Path);p.add_argument('--read-only',action='store_true',required=True);p.add_argument('--timeout',type=int,default=25);a=p.parse_args()
root=next(v for v in pathlib.Path(__file__).resolve().parents if(v/'work/hpc3_private_control.py').is_file())
s=importlib.util.spec_from_file_location('existing_direct_control',root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
cmd='python3 -c '+shlex.quote(a.script.read_text());address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
r=m.run(cmd,account='mou176',source_address=address,timeout=a.timeout)
if r.returncode==0:
 sys.stdout.buffer.write(r.stdout);sys.stderr.write('[metadata-control route=direct rc=0]\n');raise SystemExit(0)
sys.stderr.buffer.write(r.stderr);sys.stderr.write('[metadata-control direct read failed; retry same read via existing campus route]\n')
proc=subprocess.Popen([sys.executable,str(root/'work/hpc3_private_control.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
try:out,err=proc.communicate(cmd.encode(),timeout=a.timeout)
except subprocess.TimeoutExpired:
 os.killpg(proc.pid,signal.SIGKILL);out,err=proc.communicate();sys.stderr.write('[metadata-control campus read timeout; current remote state unknown]\n');sys.stderr.buffer.write(err[-1800:]);raise SystemExit(124)
sys.stdout.buffer.write(out);sys.stderr.buffer.write(err);sys.stderr.write('[metadata-control route=campus rc=%d]\n'%proc.returncode);raise SystemExit(proc.returncode)
