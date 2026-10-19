"""Five-second whole-device observations; no CUDA allocation or foreign signal."""
import argparse,json,os,subprocess,time
from pathlib import Path

def main():
 a=argparse.ArgumentParser();a.add_argument('--base',required=True);args=a.parse_args();b=Path(args.base)
 who=json.loads((b/'preparation/submit-receipt.json').read_text())['leader'];f=b/'logs/shared-resource-samples-5s.jsonl'
 with f.open('x',buffering=1)as out:
  while True:
   p=Path('/proc')/str(who['pid'])
   try:st=(p/'stat').read_text();fields=st.rsplit(')',1)[1].split();alive=int(fields[19])==who['start_ticks']and fields[0]!='Z'
   except OSError:alive=False
   row=dict(observed_unix=time.time(),leader_same_birth=alive)
   for key,query in (('gpu','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu'),('apps','--query-compute-apps=gpu_uuid,pid,used_memory')):
    try:
     result=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
     row[key]=result.stdout if result.returncode==0 else None
     row[key+'_query']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,valid=result.returncode==0)
    except subprocess.TimeoutExpired:
     row[key]=None;row[key+'_query']=dict(returncode=None,valid=False,error='NVML_QUERY_TIMEOUT')
   out.write(json.dumps(row)+'\n')
   if not alive:return
   time.sleep(5)

if __name__=='__main__':main()
