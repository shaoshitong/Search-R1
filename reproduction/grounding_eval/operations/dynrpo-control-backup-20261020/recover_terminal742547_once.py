import pathlib,json,subprocess,os,time
root=pathlib.Path('/data/user/jxu957/experiments/dynrpo-nq-grounding-20261020');old=(root/'job.id').read_text().strip();assert old=='742547'
a=subprocess.run(['sacct','-j',old,'-n','-P','--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True);assert '742547|FAILED|1:0' in a.stdout and not subprocess.check_output(['squeue','-j',old,'-h'],text=True).strip()
model=pathlib.Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4/actor/outers/outer-0200-attempt-d3eadfec8ccd440f82136d69c35cf237/huggingface')
for f in model.glob('*.safetensors'):
 with f.open('rb') as stream:assert len(stream.read(8))==8
failed=root/('failed-job-'+old);failed.mkdir()
for name in ('control','normal','job.id','submit-claim.json','submit-result.json'):
 p=root/name
 if p.exists():p.rename(failed/name)
(root/'control').mkdir()
source=root/'source/7c881cc413b916aedfe288443cd9fca838fb1c8b'
claim=dict(previous_job=old,terminal_authority=a.stdout,fix='shared_group_read0640_exact4shards',checked_unix=time.time())
with (root/'submit-claim.json').open('x') as f:json.dump(claim,f);f.flush();os.fsync(f.fileno())
cmd=['sbatch','--parsable','--output='+str(root/'control/slurm-%j.log'),'--export=ALL,GROUNDING_SOURCE='+str(source),str(source/'reproduction/grounding_eval/operations/dynrpo_grounding.sbatch')]
r=subprocess.run(cmd,capture_output=True,text=True);report=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time());(root/'submit-result.json').write_text(json.dumps(report))
if not r.returncode:
 job=r.stdout.strip().split(';')[0];assert job.isdigit();(root/'job.id').write_text(job+'\n');report['job']=job
print(json.dumps(report))
