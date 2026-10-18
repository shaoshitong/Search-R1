"""Five-second whole-device observations; no CUDA allocation or foreign signal."""
import argparse,json,os,subprocess,time
from pathlib import Path

def main():
 a=argparse.ArgumentParser();a.add_argument('--base',required=True);args=a.parse_args();b=Path(args.base)
 who=json.loads((b/'preparation/submit-receipt.json').read_text())['leader'];f=b/'logs/shared-resource-samples-5s.jsonl'
 with f.open('x',buffering=1)as out:
  while True:
   p=Path('/proc')/str(who['pid'])
   try:st=(p/'stat').read_text();alive=int(st.rsplit(')',1)[1].split()[19])==who['start_ticks']
   except OSError:alive=False
   row=dict(observed_unix=time.time(),leader_same_birth=alive,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True),
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True))
   out.write(json.dumps(row)+'\n')
   if not alive:return
   time.sleep(5)

if __name__=='__main__':main()
