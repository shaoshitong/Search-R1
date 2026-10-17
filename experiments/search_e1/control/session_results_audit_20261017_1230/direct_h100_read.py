from pathlib import Path
import importlib.util,json,sys,shlex,subprocess,datetime
D=Path(__file__).resolve().parent;target=sys.argv[1];source=D/sys.argv[2];out=D/sys.argv[3]
p=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/h1008_qprobe/management_transport.py')
s=importlib.util.spec_from_file_location('t',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
inner=m.MGMT.replace("target='h100-8'",'target='+repr(target)).replace('WORKER_CMD',repr('python3 -B -'))
cmd=['ssh','-T','-oBatchMode=yes','-oConnectTimeout=15','mgmt-host','env -u LD_LIBRARY_PATH python3 -c '+shlex.quote(inner)]
r=subprocess.run(cmd,input=source.read_text(),text=True,capture_output=True,timeout=270)
(out.with_suffix(out.suffix+'.transport.json')).write_text(json.dumps(dict(epoch=datetime.datetime.now().isoformat(),returncode=r.returncode,stderr=r.stderr)))
if r.returncode:raise RuntimeError(r.stderr[-500:])
z=json.loads(r.stdout)
if z['returncode']:raise RuntimeError(z['stderr'][-1000:])
out.write_text(z['stdout']);print(out.name,len(z['stdout']))
