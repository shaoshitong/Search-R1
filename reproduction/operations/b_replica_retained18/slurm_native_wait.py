"""Keep the actual once-only native leader inside its Slurm allocation."""
import argparse,json,time
from pathlib import Path
from reproduction.prefix_selection200.launch import submit
from reproduction.a800.owned_processes import same

def main():
 a=argparse.ArgumentParser();a.add_argument('--base',required=True);args=a.parse_args()
 b=Path(args.base);r=submit(b);print(json.dumps(r),flush=True)
 while same(r['leader']):time.sleep(3)
 if not(b/'complete.json').exists():raise RuntimeError('Native writer exited before total budget; preserve checkpoint and diagnose')

if __name__=='__main__':main()
