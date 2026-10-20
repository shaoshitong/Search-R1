import json,pathlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1');job=(b/'job.id').read_text().strip();p=b/'logs'/('train-'+job+'.log');v={'job_id':job,'observed_epoch':time.time(),'log_exists':p.exists()}
if p.is_file():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));v['log_tail']=f.read().decode(errors='replace')
for args in [['sstat','-j',job+'.batch','--noheader','--parsable2','--format=JobID,AveCPU,AveRSS,MaxRSS'],['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed']]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);v[args[0]]={'rc':r.returncode,'out':r.stdout.strip(),'stderr':r.stderr[-800:]}
print(json.dumps(v))
