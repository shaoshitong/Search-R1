import importlib.util,json,pathlib,shlex,subprocess,time
code=r'''import pathlib,json,time,subprocess
b=pathlib.Path('/data/shaoshitong/grounding-eval-20261019');pipe=b/'soft-to-hard-pipeline-v4';normal=b/'soft-to-hard-normal';cf=b/'soft-to-hard-cf-full-v4'
d=dict(checked_at=time.time(),normal_records=0,normal_bytes=0,cf_records=0,cf_bytes=0)
for root,prefix in ((normal/'records','normal'),(cf/'results','cf')):
 files=sorted(root.glob('*.json'));d[prefix+'_records']=len(files);d[prefix+'_bytes']=sum(p.stat().st_size for p in files)
 if prefix=='normal'and files:
  last=json.loads(files[-1].read_text());d['last_normal']=dict(evaluation_ordinal=last['evaluation_ordinal'],actual_batch_size=last.get('execution_batch_questions'),completed_at=last['completed_at'])
for root,name,key in ((pipe,'progress.json','pipeline_progress'),(pipe,'failed.json','pipeline_failed'),(pipe,'complete.json','pipeline_complete'),(normal,'progress.json','normal_progress'),(cf,'progress.json','cf_progress'),(cf,'verified-summary.json','verified_summary')):
 p=root/name
 if p.exists():d[key]=json.loads(p.read_text())
phase=d.get('pipeline_progress',{}).get('phase','normal');log=pipe/(phase+'.log')
if log.exists():d['current_log_tail']=log.read_text(errors='replace')[-7000:];d['current_log_bytes']=log.stat().st_size
p=pathlib.Path('/proc/967314/stat');d['pipeline_process_state']=p.read_text().split()[2]if p.exists()else None
d['gpu4']=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(d))
'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=subprocess.run(m.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=25);v=dict(transport_exit=r.returncode,stderr=r.stderr,observation=json.loads(r.stdout)if not r.returncode else r.stdout);pathlib.Path('outputs/grounding-eval-20261019/a800-soft-to-hard-v4-observation-'+str(time.time_ns())+'.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
