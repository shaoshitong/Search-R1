"""One read-only donor producer and seal-progress observation; frozen971 unchanged."""
import importlib.util,json,pathlib,subprocess,shlex,time
HERE=pathlib.Path(__file__).resolve().parent;old=json.loads((HERE/'actual-source-after466.json').read_bytes())
worker='''import pathlib,json,hashlib,time,subprocess,sys,os
B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');old=json.loads(sys.stdin.buffer.read());seals=sorted((B/'sampling').glob('shard-*/batches/batch*/complete.json'));q=t=0;latest=None
for p in seals:
 b=p.read_bytes();v=json.loads(b);assert v['run_id']=='70188b8e50fd4b45bc420ce58014f46e'and v['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28';q+=v['questions'];t+=v['trajectories'];row=dict(path=str(p.relative_to(B)),sha256=hashlib.sha256(b).hexdigest(),mtime=p.stat().st_mtime)
 if latest is None or row['mtime']>latest['mtime']:latest=row
r=subprocess.run(['systemctl','list-units','--type=service','--state=running','--all','--no-legend','--plain'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);units=[line.split()[0]for line in r.stdout.splitlines()if'70188b8e50fd'in line and'sst-q7b'in line];rows=[]
for unit in units:
 r=subprocess.run(['systemctl','show',unit,'-p','Id','-p','ActiveState','-p','SubState','-p','MainPID'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);v=dict(line.split('=',1)for line in r.stdout.splitlines()if'='in line);pid=int(v.get('MainPID','0'));v['pid_exists']=pid>0 and pathlib.Path('/proc/%d/stat'%pid).exists()
 if v['pid_exists']:
  s=pathlib.Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split();v['pid_start_ticks']=int(s[19]);v['pid_state']=s[0]
 rows.append(v)
out=dict(status='READONLY_DONOR_PRODUCTION_PROGRESS_FROZEN971_UNCHANGED',epoch=time.time(),source_node='h100-10',current_sealed_batches=len(seals),current_sealed_questions=q,current_sealed_new5_trajectories=t,previous_actual_epoch=old['epoch'],previous_sealed_batches=old['current_sealed_batches'],previous_sealed_questions=old['current_sealed_questions'],new_since_frozen971_batches=len(seals)-971,new_since_frozen971_questions=q-124194,latest_seal=latest,source_all169028_complete=q==169028,current_running_systemd_units=rows,frozen_training_cohort_questions=124194,frozen_cohort_expanded=False,producer_modified=False,payload_read=False,GPU_operation=False)
if(B/'complete.json').exists():
 b=(B/'complete.json').read_bytes();v=json.loads(b);out['source_final_complete']=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status=v.get('status'),questions=v.get('questions'),epoch=v.get('epoch'))
print(json.dumps(out))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
r=subprocess.run(ctl.route('h100-10','python3 -c '+shlex.quote(worker)),input=json.dumps(old).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
if r.returncode:raise RuntimeError('Read-only source access failed; no producer restart: '+r.stderr.decode(errors='replace')[-1000:])
x=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-production-fresh.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
