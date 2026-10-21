import subprocess,json
source=r'''import pathlib,os,json,time
out=[]
for root in pathlib.Path('/proc').iterdir():
 if not root.name.isdigit():continue
 try:
  if root.stat().st_uid!=os.getuid():continue
  args=(root/'cmdline').read_bytes().decode(errors='replace').split(chr(0))
  if'reproduction.operations.qwen7b_warmup.profile_mixed_direct_fullstep'not in args:continue
  cg=(root/'cgroup').read_text()
  if'job_746186/'not in cg:continue
  io={k:int(v)for k,v in(line.split(':')for line in(root/'io').read_text().splitlines())}
  fd=[]
  for p in(root/'fd').iterdir():
   try:
    link=os.readlink(str(p))
    if link.endswith(('.jsonl','.json','.bin','.pt','.safetensors')):fd.append(link)
   except OSError:pass
  out.append(dict(pid=int(root.name),io=io,stat_fields=(root/'stat').read_text().rsplit(')',1)[1].split()[:22],wchan=(root/'wchan').read_text(),open_feature_metadata_or_payloads=fd[-10:]))
 except (OSError,ValueError):pass
print(json.dumps(dict(epoch=time.time(),processes=out)))
'''
r=subprocess.run(['srun','--jobid=746186','--overlap','--gres=gpu:0','--ntasks=1','--cpus-per-task=1','--mem=128M','python3','-c',source],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);print(r.stdout)
if r.returncode:raise RuntimeError(r.stderr[-1200:])
