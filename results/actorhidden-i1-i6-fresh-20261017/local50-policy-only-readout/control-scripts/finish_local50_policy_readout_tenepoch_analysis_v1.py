from pathlib import Path
import datetime,hashlib,json,base64
from reproduction.h1008_qprobe.established_controller_transport import remote
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');repo=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016');state=json.loads((root/'work/local-probe-fit50-policy-readout-tenepoch-state.json').read_text());tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');summary_code=(repo/'results/actorhidden-i1-i6-fresh-20261017/local50_fixed_tenepoch_analysis.py').read_text();plot_code=(repo/'results/actorhidden-i1-i6-fresh-20261017/plot_local50_fixed_tenepoch.py').read_text();artifact=root/'outputs/local-probe50-policy-only-readout-20261017';artifact.mkdir(parents=True,exist_ok=True)
inner='''import pathlib,json,hashlib,base64
b=pathlib.Path(BASE);assert (b/'logs/pipeline-complete.json').exists()
ns={};exec(compile(ANALYSIS,'published_fixed_analysis','exec'),ns)
result=ns['inspect'](b,b/'repo/results/actorhidden-i1-i6-fresh-20261017')
out=b/'preparation/local50-final-analysis';out.mkdir(exist_ok=True);summary=out/'aggregate-metrics.json';summary.write_text(json.dumps(result,indent=2)+'\\n')
plots={};exec(compile(PLOT,'scientific_plot_control','exec'),plots);plots['render'](summary,out/'fixed-data-learning-curves')
config=json.loads((b/'preparation/launch-config.json').read_text());baseline=pathlib.Path(config['local_probe_fit50']['fixed_data_source_base'])/'preparation/local50-final-analysis/aggregate-metrics.json';plots['render_comparison'](baseline,summary,out/'question-readout-ablation-comparison')
files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),base64=base64.b64encode(p.read_bytes()).decode())for p in out.iterdir()if p.suffix in ('.png','.pdf')}
print(json.dumps(dict(summary=result,remote_directory=str(out),summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),figures=files)))
'''.replace('BASE',repr(state['base'])).replace('ANALYSIS',repr(summary_code)).replace('PLOT',repr(plot_code))
script='import subprocess,os\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2");r=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=env,text=True,capture_output=True,timeout=180);assert r.returncode==0,r.stderr;print(r.stdout)\n'
value=json.loads(remote(script,str(root/'work'/('local50-policy-final-analysis-'+tag+'.transport.json'))));summary=artifact/'aggregate-metrics.json';summary.write_text(json.dumps(value['summary'],indent=2)+'\n');assert hashlib.sha256(summary.read_bytes()).hexdigest()==value['summary_sha256']
for name,metadata in value['figures'].items():
 data=base64.b64decode(metadata.pop('base64'));assert len(data)==metadata['bytes']and hashlib.sha256(data).hexdigest()==metadata['sha256'];(artifact/name).write_bytes(data)
print(json.dumps(dict(remote_directory=value['remote_directory'],local_summary=str(summary),summary_sha256=value['summary_sha256'],figures=value['figures']),indent=2));print('epoch trainBA testBA trainTraj testTraj trainTotal testTotal')
for e in value['summary']['epochs']:
 t=e['metrics']['train'];v=e['metrics']['test'];print(e['epoch'],t['balanced_accuracy'],v['balanced_accuracy'],t['trajectory_accuracy'],v['trajectory_accuracy'],t['total_loss'],v['total_loss'])
