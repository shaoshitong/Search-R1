import importlib.util,json
from pathlib import Path
p=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py');s=importlib.util.spec_from_file_location('bridge',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);D=Path(__file__).resolve().parent;r=m.run('mou176','python3 -B -',(D/'read_topd_ancestors.py').read_text(),'/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work');
if r.returncode:print(r.stderr[-900:])
else:(D/'topd-ancestors.json').write_text(r.stdout);print('saved',len(r.stdout))
