import pathlib,json,subprocess,time
a=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');suite=a/'online-mixed-layer-prefix-20261022-2gpu-v1';accepted=a/'online-mixed-pilot-CPU-acceptance-20261022-v1';rows=[]
def read(p):return json.loads(p.read_bytes())
for layer,prefix in((27,20),(21,20),(27,80),(21,80),(15,20),(18,20),(24,20)):
 name='layer%d-prefix%d'%(layer,prefix);b=suite/name;r=dict(variant=name,complete_exists=(b/'complete.json').exists(),accepted_exists=(accepted/name/'accepted.json').exists())
 if(b/'root-submission.json').exists():r['job_id']=read(b/'root-submission.json')['job_id']
 for key in('complete.json','progress-rank-0.json','progress-rank-1.json'):
  if(b/key).exists():
   v=read(b/key);r[key]={k:v[k]for k in('status','stage','epochs','epoch','cursor','rank','epoch_questions_seen','time')if k in v}
 if r['accepted_exists']:
  v=read(accepted/name/'accepted.json');p=pathlib.Path(v['checkpoint']['path']);s=p.stat();actual=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns);r['accepted_checkpoint_metadata_unchanged']=all(v['file_stat'][k]==actual[k]for k in('inode','bytes','mtime_ns','ctime_ns'));r['accepted_SHA']=v['checkpoint']['sha256'];r['accepted_status']=v['status']
 rows.append(r)
jobs=','.join(r['job_id']for r in rows if'job_id'in r)
cmd=['sacct','-X','-n','-P','-j',jobs,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'];raw=subprocess.check_output(cmd,universal_newlines=True,timeout=15);states={v.split('|')[0]:v.split('|')for v in raw.splitlines()if'|'in v}
for r in rows:
 state=states.get(r.get('job_id'),[]);r['sacct']=state;r['eligible_new_CPU_acceptance']=r['complete_exists']and len(state)>2 and state[1:3]==['COMPLETED','0:0']and not r['accepted_exists']
print(json.dumps(dict(time=time.time(),variants=rows,final_read=False,Slurm_mutations=False)))
