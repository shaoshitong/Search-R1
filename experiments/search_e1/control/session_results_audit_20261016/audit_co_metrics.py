from pathlib import Path
import sys,json,importlib.util,datetime
D=Path(__file__).resolve().parent
bp='/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py'
spec=importlib.util.spec_from_file_location('bridge',bp);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.run('jxu957','python3 -B -',(D/'read_co_metrics_remote.py').read_text(),'/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work',login_host='10.120.48.31',observation_timeout=120,expected_host_identity={'hostname':'ACD-Login-1','boot_id':'a27512a3-24a0-4ca3-aa92-98d2a5256d9a','pid_namespace':'pid:[4026531836]'})
try:d=json.loads(r.stdout)
except Exception:d={'error':r.stderr[-2000:],'stdout':r.stdout[-1000:]}
d['transport_returncode']=r.returncode
p=D/('co-observation-'+sys.argv[1]+'.json');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(p),'returncode':r.returncode,'utc':d.get('utc'),'runs':[{'root':x['root'],'complete':x.get('completed_iter'),'training':len(x['train']),'fixed':x['fixed512'],'files':[(p['path'],list(p['data']) if isinstance(p['data'],dict) else '') for p in x['files']]} for x in d.get('runs',[])]},ensure_ascii=False))
