import pathlib,json,subprocess,time
suite=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-loss128-v2');a=json.loads((suite/'preparation.json').read_text());pub=json.loads((suite/'source-publication.json').read_text());out=dict(time=time.time(),source_commit=pub['commit'],variants={})
for name,v in a['variants'].items():
 b=pathlib.Path(v['base']);d=dict(base=str(b));p=b/'job.id'
 if p.exists():
  job=p.read_text().strip();d['job_id']=job;d['queue']=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R'],universal_newlines=True,timeout=15).strip();d['accounting']=subprocess.check_output(['sacct','-n','-P','-j',job,'-o','JobID,State,ExitCode,Elapsed,NodeList'],universal_newlines=True,timeout=15).strip()
 for f in ['controller.json','progress.json','latest.json','next-update-replay.json','failure.json','metrics.json','complete.json']:
  if (b/f).exists():d[f]=json.loads((b/f).read_text())
 logs=sorted((b/'logs').glob('fit-*.log'),key=lambda p:p.stat().st_mtime)
 if logs:
  p=logs[-1];d['log_bytes']=p.stat().st_size
  with p.open('rb') as h:h.seek(max(0,p.stat().st_size-3500));d['log_tail']=h.read().decode(errors='replace')
 out['variants'][name]=d
print(json.dumps(out))
