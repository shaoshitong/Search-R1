"""Root-owned, staged development-only order audit; preserve ambiguous receipts."""
import pathlib,json,hashlib,fcntl,subprocess,re,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/packed-Z-E8-trajectory-permutation-audit-20261023-v1')
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if p.returncode:raise RuntimeError(p.stdout[-800:]+p.stderr[-1000:])
 return p.stdout.strip()
lock=(b/'root-submit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);entry=b/'audit.sbatch';assert proof(entry)['sha256']=='f028372b9c94b43f425a246a72bb8e153529eace685f861baeb32d9c62813e78'
assert proof(b/'audit-config.json')['sha256']=='41fbd8e36070501a257fd068f41615f05d09f0ee50d3ebfc8736b64daabced16'and proof(b/'source-publication.json')['sha256']=='b5c6a8dd54426d8a897094e205bf86e7205acfc539724bb4470388acb55b3d45'
receipt=b/'root-submission.json';intent=b/'root-submit-intent.json'
if receipt.exists():print(json.dumps(read(receipt)))
else:
 name=re.search(r'^#SBATCH --job-name=(.+)$',entry.read_text(),re.M).group(1);rows=run(['squeue','-h','-u','mou176','-n',name,'-o','%i']).splitlines()
 if rows:
  assert len(rows)==1;job=rows[0];details=run(['scontrol','show','job',job,'-o']);assert 'Command='+str(entry)in details and 'UserId=mou176('in details
 else:
  if intent.exists():raise RuntimeError('Previous submit outcome unknown; inspect same name/accounting before any retry.')
  intent.write_text(json.dumps({'time':time.time(),'entry':proof(entry),'name':name},sort_keys=True)+'\n')
  job=run(['sbatch','--parsable','--exclude=ACD1-13',str(entry)]).split(';')[0];assert job.isdigit()
 result={'time':time.time(),'job_id':job,'name':name,'entry':proof(entry),'config':proof(b/'audit-config.json'),'source_commit':'58a6cf770ed99fd6326fc45d3fcd68099cd239b0','checkpoint_sha256':'2d81f15cd4af00fe192963cba6f6db86aba22bf47523bd3e10982b86f7178483','GPUs':2,'partition':'acd_u','time_limit':'00:25:00','fixed_development_G5':True,'final4096_read':False,'no_dependencies':True}
 receipt.write_text(json.dumps(result,sort_keys=True)+'\n');print(json.dumps(result))
