# Historical exact operator from2026-10-22. No credentials; do not replay rejected CPU-only sbatch.
import sys,json,shlex,time
from pathlib import Path
w=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6');sys.path.insert(0,str(w/'work/easy7b-fa3-fastpath-20261022'))
from reproduction.operations.easy7b_20261021 import watch_cp34_recovery_local as wc
b=wc.shared.load_bridge(w)
code=Path(sys.argv[1]).read_text()
r=b.run('jxu957',shlex.join(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code]),'',str(w/'work'),login_host='10.120.48.31',expected_host_identity={'hostname':'ACD-Login-1','boot_id':'a27512a3-24a0-4ca3-aa92-98d2a5256d9a','pid_namespace':'pid:[4026531836]'},observation_timeout=50)
p=w/'outputs/easy2560-qwen7b-20261021'/(sys.argv[2]+'-'+str(time.time_ns())+'.json');p.write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');print(p)
print(r.stdout);print(r.stderr[-2000:]);sys.exit(r.returncode)
