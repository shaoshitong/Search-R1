"""User-authorized same-card native recovery; keep foreign processes intact."""
import argparse,json,os,socket,subprocess,time
from pathlib import Path
from reproduction.prefix20_online.launch import read,once,publish,sha
from reproduction.prefix_selection200.launch import submit
from reproduction.a800.owned_processes import same

def main():
 a=argparse.ArgumentParser();a.add_argument('--base',required=True);args=a.parse_args();b=Path(args.base)
 initial=read(b/'preparation/launch-config.json')
 node=socket.gethostname()
 assert node in ('h100-5','h100-8','h100-10','h100-11') and node==initial['expected_execution_node']
 assert not(b/'preparation/submit-intent.json').exists(),'Reconcile an existing native submission instead'
 q=subprocess.check_output(['squeue','-h','-n','sst-b-diverse-cp64-eight','-o','%T'],text=True).strip()
 assert not q or q=='CANCELLED','Own queued execution must be cancelled before native launch'
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
 inventory=[line.split(',')for line in rows.splitlines()if int(line.split(',')[0])in initial['physical_gpu_indices']];assert len(inventory)==initial['world_size']
 cp=b/'preparation/launch-config.json';c=read(cp);before=sha(cp);uuids=[r[1].strip()for r in inventory]
 c.update(physical_gpu_uuids=uuids,auxiliary_gpu_uuids=uuids)
 once(b/'preparation/shared-hardware-dispatch.json',dict(node=node,observed_unix=time.time(),before_config_sha256=before,inventory=inventory,slurm_reserved_gpus=0,shared_existing_foreign_jobs_authorized=True,foreign_processes_stopped=False,actor_probe_optimizer_modified=False))
 publish(cp,c);receipt=submit(b);print(json.dumps(receipt),flush=True)
 # The service must retain a live main process; detached session IDs alone
 # do not protect grandchildren from systemd cgroup cleanup on main exit.
 while same(receipt['leader']):
  finished,_=os.waitpid(receipt['leader']['pid'],os.WNOHANG)
  if finished:break
  time.sleep(3)
 final=b/'logs/pipeline-complete.json'
 if not final.exists() or read(final).get('complete_outers')!=200:
  raise RuntimeError('Native writer exited; preserve actual state and diagnose')

if __name__=='__main__':main()
