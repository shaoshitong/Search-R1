import pathlib,json,subprocess,hashlib,time
entry=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-window-terminal-CPU-acceptance-20261023-v1/code-76637b36e33a/current64.sh');assert hashlib.sha256(entry.read_bytes()).hexdigest()=='3e5c4c0626feee8cf3bc5b6b1b0b4e2d072674def5fbd8d697571ebe21bd8aea'
p=subprocess.run(['bash',str(entry)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=55);out={'time':time.time(),'returncode':p.returncode,'stdout':p.stdout[-3000:],'stderr':p.stderr[-3000:]}
accepted=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-window-terminal-CPU-acceptance-20261023-v1/current64/epoch-10/accepted.json')
if accepted.is_file():out['accepted']=json.loads(accepted.read_bytes())
print(json.dumps(out))
