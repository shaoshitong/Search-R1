from pathlib import Path
import json,subprocess,sys,concurrent.futures,shlex
D=Path(__file__).resolve().parent
H='/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/read_search_g1_shared3381_direct_v1.py'
C='/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/cosearch_mou176_control.py'
roots={'gigpo':['/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs/gigpo-e1-fresh200-live-job3347-1791494104474180576','/mnt/data/shaoshitong/search-g1-qprobe3-stdloss-545d4cf-20261015-v2','/mnt/data/shaoshitong/search-g1-qprobe3-resume5-fb68e88-20261015'],'co':['/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015','/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014']}
def run(name):
 c='import sys\nsys.argv=[\"read\",'+repr(json.dumps(roots[name]))+']\n'+(D/'read_gico_targeted.py').read_text()
 payload=c if name=='gigpo' else json.dumps({'command':"python3 -B - <<'REMOTE'\n"+c+'\nREMOTE'})
 
 if name=='co':
  payload=c
  args=['python3','/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py','--account','jxu957','--control-work','/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work','--command','python3 -B -']
 else:args=['python3',H]
 p=subprocess.run(args,input=payload,capture_output=True,text=True,timeout=90,cwd=str(Path(H).parent.parent))
 try:d=json.loads(p.stdout) if p.returncode==0 else {'error':p.stderr[-2000:],'returncode':p.returncode}
 except Exception as e:d={'error':str(e),'stdout':p.stdout[-1500:],'stderr':p.stderr[-1500:]}
 (D/(name+'-targeted-'+sys.argv[1]+'.json')).write_text(json.dumps(d));print(name,p.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['gigpo','co']))
