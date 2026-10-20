import importlib.util,subprocess,shlex,pathlib,json,time
root=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
s=importlib.util.spec_from_file_location('c',root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
a=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
c='''import json,pathlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-20261020');job=(b/'job.id').read_text().strip()
out={'time':time.time(),'job_id':job}
for key,args in [('queue',['squeue','-h','-j',job,'-o','%i|%T|%P|%M|%N|%R']),('accounting',['sacct','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed,AllocTRES'])]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);out[key]=r.stdout+r.stderr
for name in ['restore-acceptance.json','deterministic-warmup.json','next-update-replay-diagnostics-rank0.json','next-update-replay-diagnostics-rank1.json','next-update-replay-diagnostics-rank2.json','next-update-replay-diagnostics-rank3.json','next-update-proof.json','prefix20-initialization-proof.json','progress.json','latest.json','metrics-latest.json','complete.json','phase.json']:
 p=b/'probe'/name
 if p.exists():
  data=json.loads(p.read_text())
  if name=='restore-acceptance.json':data['source_checkpoint']={k:data['source_checkpoint'][k] for k in ('path','sha256','steps','epoch')}
  if name=='latest.json':data={k:v for k,v in data.items() if k!='metrics'}
  out[name]=data
p=b/'logs'/('fit-'+job+'.log')
if p.exists():out['log_bytes']=p.stat().st_size;out['log_tail']=p.read_text()[-6500:]
print(json.dumps(out))
'''
publication=json.loads((root/'outputs/qwen7b-warmup-20261019/mixed-ft-source-publication.json').read_text())
c=c.replace('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-20261020',publication['base'])
r=m.run('python3 -c '+shlex.quote(c),account='mou176',source_address=a,timeout=45)
print('returncode',r.returncode)
for line in r.stdout.decode().splitlines():
 if line.startswith('{'):
  data=json.loads(line);d=root/'outputs/qwen7b-warmup-20261019';(d/'mixed-ft-latest-status.json').write_text(json.dumps(data,indent=2));(d/('mixed-ft-status-%d.json'%time.time())).write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
 else:print(line)
if r.returncode:print(r.stderr.decode())
