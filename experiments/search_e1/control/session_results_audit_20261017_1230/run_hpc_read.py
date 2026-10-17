import importlib.util,json,sys
from pathlib import Path
p=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py');s=importlib.util.spec_from_file_location('bridge',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
D=Path(__file__).resolve().parent;account=sys.argv[1];code='ACCOUNT='+repr(account)+'\n'+(D/'read_hpc_scalar.py').read_text();r=m.run(account,'python3 -B -',code,'/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work',observation_timeout=55)
(D/(account+'-current.transport.json')).write_text(json.dumps(dict(returncode=r.returncode,stderr=r.stderr),indent=2)+'\n')
if r.returncode:print(account,'read failed',r.returncode,r.stderr[-1600:])
else:(D/(account+'-current.json')).write_text(r.stdout);print(account,'read saved',len(r.stdout))
