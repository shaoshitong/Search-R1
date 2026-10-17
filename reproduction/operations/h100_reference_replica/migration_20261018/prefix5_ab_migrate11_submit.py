# Archived actual operation. Reconcile existing intent/receipt before any use.
exec(open('work/prefix5_ab_migrate11_inspect.py').read().split('with concurrent.futures.ThreadPoolExecutor')[0])
prepared=json.loads(Path('work/prefix5-ab-migrate11-prepared-arms.json').read_text());commit='2e3976230ac9425d66d8553afa9962821e964932'
wrapper='/mnt/shared/workspace/training_data_node_8/shaoshitong/search-g1-prefix5-reference-preparation-20261018/code/'+commit+'/repo/reproduction/operations/h100_reference_replica/h10011_private_paths.sh'
inner='import sys,json\nARMS='+repr(prepared)+'\nrepo=next(iter(ARMS.values()))["base"]+"/repo"\nsys.path.insert(0,repo)\nfrom reproduction.prefix_selection200.launch import submit\nprint(json.dumps({s:submit(v["base"])for s,v in ARMS.items()}))'
script='import subprocess,json,os\np=subprocess.run(["unshare","--mount","--propagation","private","bash",'+repr(wrapper)+',"/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),text=True,capture_output=True,timeout=120)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
p=subprocess.run(ctl.route('h100-11','/usr/bin/python3 -'),input=script,text=True,capture_output=True,timeout=150)
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');Path('work/prefix5-ab-migrate11-submit-'+tag+'.json').write_text(json.dumps({'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2));assert p.returncode==0,p.stderr
r=json.loads(p.stdout);Path('work/prefix5-ab-migrate11-submit-receipts.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
Path('work/prefix5-ab-pre-migrate11-arms.json').write_text(Path('work/prefix5-ab-assigned-arms.json').read_text());Path('work/prefix5-ab-assigned-arms.json').write_text(json.dumps(prepared,indent=2))
