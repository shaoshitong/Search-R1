import pathlib,json,subprocess,datetime,os
P=pathlib.Path;base=P('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v2')
def once(p,x):
 with P(p).open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
assert (base/'preparation/stop3381-receipt.json').exists()
assert not (base/'preparation/submit-intent.json').exists(),'Inspect intent/receipt, never replay submission'
q=subprocess.check_output(['squeue','-h','-w','h100-8','-o','%i'],text=True).strip();assert not q,'Preserve other active jobs; do not submit overlapping allocation'
cmd=['sbatch','--exclusive','--parsable','--job-name=search-g1-final200-bs256','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=96','--mem=512G','--gres=gpu:8','--time=3-00:00:00','--signal=B:TERM@60','--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),'--export=NONE',str(base/'preparation/launch-final.sh')]
once(base/'preparation/submit-intent.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'scope':'Final51713 only; CP200/Bs256/cache.35; zero additional training'})
r=subprocess.run(cmd,capture_output=True,text=True);v={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};once(base/'preparation/submit-receipt.json',v);assert r.returncode==0;print(json.dumps(v))
