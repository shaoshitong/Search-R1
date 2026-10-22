import json,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):return json.loads(Path(p).read_bytes())
d=read(A/'donor14080-independent-L26-prepared-v1/bank/groups.json')['train'];g=d[0];p=read(A/'hpc3-tail-newseedG5-20261021-v1/sampling/shard-01/batches/batch-000420/complete.json');print(json.dumps(dict(donor_example=g,seal=p,final4096_read=False)))
