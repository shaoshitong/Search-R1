import argparse,importlib.util,json,shlex,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('script',type=Path);p.add_argument('--timeout',type=int,default=45);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('control',root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
r=m.run('python3 -c '+shlex.quote(a.script.read_text()),account='mou176',source_address=address,timeout=a.timeout)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)
