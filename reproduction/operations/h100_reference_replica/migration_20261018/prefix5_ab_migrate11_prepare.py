# Archived actual operation. Reconcile existing intent/receipt before any use.
exec(open('work/prefix5_ab_migrate11_inspect.py').read().split('with concurrent.futures.ThreadPoolExecutor')[0])
x=json.loads(json.loads(sorted(Path('work').glob('prefix5-ab-migrate11-target-inspect-*.json'))[-1].read_text())['stdout'])
import csv
uuids=[r[1].strip()for r in csv.reader(x['gpu']['stdout'].splitlines())]
stage=json.loads(Path('work/prefix5-reference-code-stage.json').read_text());assert stage['commit']=='2e3976230ac9425d66d8553afa9962821e964932'
inner='import sys,json\nsys.path.insert(0,'+repr(stage['repo'])+')\nfrom reproduction.prefix_selection200.node_migration import prepare_first\nARMS='+repr(arms)+'\nprint(json.dumps({s:prepare_first(v["base"],'+repr(stage['repo'])+','+repr(stage['commit'])+',s,'+repr(uuids)+')for s,v in ARMS.items()}))'
script='import subprocess,os\np=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),text=True,capture_output=True,timeout=150)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
p=subprocess.run(ctl.route('h100-8','/usr/bin/python3 -'),input=script,text=True,capture_output=True,timeout=180)
v={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr};Path('work/prefix5-ab-migrate11-prepare-result.json').write_text(json.dumps(v,indent=2));assert p.returncode==0,v
prepared=json.loads(p.stdout);Path('work/prefix5-ab-migrate11-prepared-arms.json').write_text(json.dumps(prepared,indent=2));print(json.dumps({s:{k:v[k]for k in ['base','run_id','node','global_micro','local_micro','accumulation','files']}for s,v in prepared.items()}))
