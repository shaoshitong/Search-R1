import pathlib,json,subprocess,time,os
root=pathlib.Path('/data/user/jxu957/experiments/dynrpo-nq-grounding-20261020');source=root/'source/7c881cc413b916aedfe288443cd9fca838fb1c8b';claim=root/'submit-claim.json'
assert not claim.exists() and not (root/'job.id').exists()
with claim.open('x') as f:json.dump(dict(source=str(source),requested_GPU=1,partition='emergency_acd',checked_unix=time.time()),f);f.flush();os.fsync(f.fileno())
cmd=['sbatch','--parsable','--output='+str(root/'control/slurm-%j.log'),'--export=ALL,GROUNDING_SOURCE='+str(source),str(source/'reproduction/grounding_eval/operations/dynrpo_grounding.sbatch')]
r=subprocess.run(cmd,capture_output=True,text=True);receipt=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time());(root/'submit-result.json').write_text(json.dumps(receipt))
if not r.returncode:
 job=r.stdout.strip().split(';')[0];assert job.isdigit();(root/'job.id').write_text(job+'\n');receipt['job']=job
print(json.dumps(receipt))
