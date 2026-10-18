"""Keep the actual once-only native leader inside its Slurm allocation."""
import argparse,json,time,os,socket,subprocess
from pathlib import Path
from reproduction.prefix_selection200.launch import submit
from reproduction.a800.owned_processes import same
from reproduction.prefix20_online.launch import read,once,publish,sha

def main():
 a=argparse.ArgumentParser();a.add_argument('--base',required=True);args=a.parse_args()
 b=Path(args.base);cp=b/'preparation/launch-config.json'
 if not(b/'preparation/submit-intent.json').exists():
  node=socket.gethostname();assert node in ('h100-8','h100-10')
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
  inventory=[line.split(',')for line in rows.splitlines()];assert len(inventory)==8
  assert all(int(row[3])-int(row[2])>=45000 for row in inventory),'Allocated cards lack the unchanged startup memory budget'
  c=read(cp);before=sha(cp);uuids=[row[1].strip()for row in inventory]
  c.update(expected_execution_node=node,physical_gpu_uuids=uuids,auxiliary_gpu_uuids=uuids)
  once(b/'preparation/slurm-hardware-dispatch.json',dict(node=node,job_id=os.environ['SLURM_JOB_ID'],before_config_sha256=before,inventory=inventory,actor_probe_optimizer_modified=False))
  publish(cp,c)
 r=submit(b);print(json.dumps(r),flush=True)
 while same(r['leader']):time.sleep(3)
 if not(b/'complete.json').exists():raise RuntimeError('Native writer exited before total budget; preserve checkpoint and diagnose')

if __name__=='__main__':main()
