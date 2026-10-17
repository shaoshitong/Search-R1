from pathlib import Path
import datetime,json,hashlib
from reproduction.h1008_qprobe.established_controller_transport import remote
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');state=json.loads((root/'work/local-probe-fit50-tenepoch-state.json').read_text());tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
inner='''import pathlib,sys,json,importlib.util
b=pathlib.Path(BASE);sys.path.insert(0,str(b/'repo'));path=b/'repo/results/actorhidden-i1-i6-fresh-20261017/local50_identical_input_analysis.py';spec=importlib.util.spec_from_file_location('local50_conflicts',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);r=b/'outers/outer-000051';d=json.loads((r/'diagnostic-complete.json').read_text());scan=pathlib.Path(d['scan_path']);result=dict(train=m.inspect(r/'prefix/scan',r/'full/own-em.jsonl'),test=m.inspect(scan,scan.parent.parent/'full/own-em.jsonl'));print(json.dumps(result))
'''.replace('BASE',repr(state['base']))
script='import subprocess,os\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2");r=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=env,text=True,capture_output=True,timeout=120);assert r.returncode==0,r.stderr;print(r.stdout)\n'
value=json.loads(remote(script,str(root/'work'/('local50-fixed-input-conflicts-'+tag+'.transport.json'))));out=root/'work'/('local50-fixed-input-conflicts-'+tag+'.json');out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value));print('proof',str(out),'SHA',hashlib.sha256(out.read_bytes()).hexdigest())
