import sys,json,re,shlex,importlib.util,time
from pathlib import Path
w=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6');job=sys.argv[1]
spec=importlib.util.spec_from_file_location('campus_ctl',w/'work/start_cosearch_flash_hf.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
secret=Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text();line=next(l for l in secret.splitlines() if l.startswith('账号 jxu957'));password=re.split(r'密码\s*[:：]?\s*',line,1)[1].strip().strip('`');secret=line=None
code="""import sys,json,importlib.util
from pathlib import Path
p=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/source/b8df9cc12904ae6729768107ff4aa529cff666af/reproduction/operations/easy7b_20261021/fa3_archive_consumers_hk.py')
spec=importlib.util.spec_from_file_location('new_consumers',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
payload=json.load(sys.stdin)
r=m.launch('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023','2d5183455551145c52d1783a38663e2fcf170ceb',payload['job'],payload['password'],'b8df9cc12904ae6729768107ff4aa529cff666af',first_step=100,login_host='10.120.48.31',login_identity_json=json.dumps({'hostname':'ACD-Login-1','boot_id':'a27512a3-24a0-4ca3-aa92-98d2a5256d9a','pid_namespace':'pid:[4026531836]'}))
print(json.dumps({'status':r['status'],'job':r['source_job'],'requested_steps':r['requested_steps'],'core':r['core']['birth'],'companion':r['companion']['birth']}),flush=True)
"""
r=m.campus('hkust3',shlex.join(['python3','-B','-c',code]),input=json.dumps({'password':password,'job':job}),observation_timeout=55);password=None
p=w/'outputs/easy2560-qwen7b-20261021'/('fastpath-leanmetrics-r1-hk-archive-launch-'+str(time.time_ns())+'.json');p.write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');print(p);print(r.stdout);print(r.stderr[-2000:]);sys.exit(r.returncode)
