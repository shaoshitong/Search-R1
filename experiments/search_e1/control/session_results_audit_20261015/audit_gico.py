from pathlib import Path
import json,subprocess,sys,concurrent.futures
D=Path(__file__).resolve().parent
H='/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/read_search_g1_shared3381_direct_v1.py'
C='/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/cosearch_mou176_control.py'
def run(name):
 c=(D/('read_'+name+'_remote.py')).read_text()
 args=['python3',H if name=='gigpo' else C]
 payload=c if name=='gigpo' else json.dumps({'command':"python3 -B - <<'REMOTE'\n"+c+'\nREMOTE'})
 p=subprocess.run(args,input=payload,capture_output=True,text=True,timeout=90,cwd=str(Path(H).parent.parent))
 out=D/(name+'-actual-'+sys.argv[1]+'.json')
 try:d=json.loads(p.stdout) if p.returncode==0 else {'error':p.stderr[-1500:],'returncode':p.returncode}
 except Exception as e:d={'error':str(e),'stdout':p.stdout[-1500:],'stderr':p.stderr[-1500:]}
 out.write_text(json.dumps(d,ensure_ascii=False));print(name,p.returncode,'saved',out.name,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['gigpo','co']))
