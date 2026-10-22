import runpy,pathlib,json,subprocess,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1')
module=runpy.run_path(str(base/'prepare_frozen_final4096_generation.py'))
x=module['status']();x['epoch']=time.time()
x['squeue']=subprocess.check_output(['squeue','-h','-u','mou176','-n','q7b-final-cA,q7b-final-cB,q7b-final-sA,q7b-final-sB','-o','%i|%T|%j|%N|%R'],universal_newlines=True).strip().splitlines()
print(json.dumps(x))
