import pathlib,hashlib
p=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1/production-complete-blind.json')
raw=p.read_bytes();assert len(raw)==131101 and hashlib.sha256(raw).hexdigest()=='ffdbc91f0bd34a1e5f51dd1a764850099f4daa5ffe6fc416ee189e56d86bff6f'
print(raw.decode(),end='')
