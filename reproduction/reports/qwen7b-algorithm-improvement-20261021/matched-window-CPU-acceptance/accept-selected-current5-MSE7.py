import pathlib,json,subprocess,hashlib,time
entries=[({'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-window-terminal-CPU-acceptance-20261023-v1/code-76637b36e33a/current64.sh', 'bytes': 1056, 'sha256': '3e5c4c0626feee8cf3bc5b6b1b0b4e2d072674def5fbd8d697571ebe21bd8aea'}, 5), ({'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-window-terminal-CPU-acceptance-20261023-v1/code-76637b36e33a/mse-M5.sh', 'bytes': 1017, 'sha256': 'b19611693ac2ae444b0d9ca62651e3559337123d7babcfc5b7d3354b1ee54a0d'}, 7)]
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
for ref,epoch in entries:
 entry=pathlib.Path(ref['path']);assert hashlib.sha256(entry.read_bytes()).hexdigest()==ref['sha256'];p=subprocess.run(['bash',str(entry),'--epoch',str(epoch)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=55);out={'time':time.time(),'entry':ref,'epoch':epoch,'returncode':p.returncode,'stdout':p.stdout[-3000:],'stderr':p.stderr[-3000:]}
 accepted=A/'matched-window-terminal-CPU-acceptance-20261023-v1'/entry.stem/('epoch-%02d'%epoch)/'accepted.json'
 if accepted.exists():out['accepted']=json.loads(accepted.read_bytes())
 print(json.dumps(out),flush=True)
 if p.returncode:break
