import pathlib,json,subprocess,hashlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');entry=A/'matched-window-terminal-CPU-acceptance-20261023-v1/code-c925b5cf54f2/current64.sh';assert hashlib.sha256(entry.read_bytes()).hexdigest()=='7c0f9a0641b6be82fe0eaec4d2f92b0fe240388aa3d4b7eba1e5e52763a2c279'
p=subprocess.run(['bash',str(entry)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=55);out={'time':time.time(),'returncode':p.returncode,'stdout':p.stdout[-3000:],'stderr':p.stderr[-3000:]}
accepted=A/'matched-window-terminal-CPU-acceptance-20261023-v1/current64/epoch-10/accepted.json'
if accepted.is_file():out['accepted']=json.loads(accepted.read_bytes())
print(json.dumps(out))
