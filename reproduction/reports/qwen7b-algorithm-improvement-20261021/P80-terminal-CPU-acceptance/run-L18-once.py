import subprocess,json
code='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-P80-terminal-CPU-acceptance-20261023-v1/code-b3a3917b4d7a'
p=subprocess.run(['bash',code+'/acceptance.sh','--layer','18','--source-sha','0a38f57a27f096b1bc1881a89bda121642f49eccbf3c45249b68508fd66b58a4'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=50)
if p.returncode:raise RuntimeError(p.stdout[-1200:]+p.stderr[-2200:])
print(p.stdout.strip())
