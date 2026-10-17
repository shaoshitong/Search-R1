from pathlib import Path
import json,sys,hashlib,datetime
repo=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016');sys.path.insert(0,str(repo))
from reproduction.h1008_qprobe.established_controller_transport import remote
base='/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-dataset-20261017-0bdddc20840b-r48'
sources={name:(repo/'reproduction/prefix_selection200'/name).read_text()for name in('probe20k_fit.py','probe20k_cpu_checks.py')}
sources['replay_train_epoch.py']=(repo/'reproduction/prefix20_question_replay/train_epoch.py').read_text()
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');dest=base+'/preparation/fit-cpu-controls/'+tag
script='''import pathlib,json,sys,os,subprocess,hashlib
root=pathlib.Path(DEST);root.mkdir(parents=True,exist_ok=False)
sources=SOURCES
for name,code in sources.items():(root/name).write_text(code)
code="""import sys,importlib.util,json\nsys.path.insert(0,FROZEN_REPO)\nimport reproduction.prefix_selection200\nfor short in ('replay_train_epoch','probe20k_fit','probe20k_cpu_checks'):\n p=ROOT/(short+'.py');name='reproduction.prefix20_question_replay.train_epoch'if short=='replay_train_epoch'else'reproduction.prefix_selection200.'+short\n spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)\nprint(json.dumps(module.run()))\n""".replace('FROZEN_REPO',repr(BASE+'/repo')).replace('ROOT',repr(str(root)))
code=code.replace("p="+repr(str(root))+"/", "p=__import__('pathlib').Path("+repr(str(root))+")/")
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
r=subprocess.run(['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',code],env=env,capture_output=True,text=True,timeout=90)
value=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,source_sha256={name:hashlib.sha256(code.encode()).hexdigest()for name,code in sources.items()},cpu_only=True)
(root/'RESULT.json').write_text(json.dumps(value,indent=2)+'\\n');print(json.dumps(value))
'''.replace('DEST',repr(dest)).replace('SOURCES',repr(sources)).replace('BASE',repr(base))
path=Path('work')/('probe20k-cpu-fixtures-'+tag+'.transport.json');value=remote(script,str(path));out=path.with_name(path.name.replace('.transport',''));out.write_text(json.dumps(json.loads(value),indent=2)+'\n');print(json.dumps(dict(actual=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),result=json.loads(value))))
