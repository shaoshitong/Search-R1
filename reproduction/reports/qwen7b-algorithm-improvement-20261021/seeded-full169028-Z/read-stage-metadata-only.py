from pathlib import Path
import json,time
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1')
print(json.dumps(dict(time=time.time(),base_exists=b.is_dir(),files=[dict(path=str(p),bytes=p.stat().st_size)for p in b.rglob('*')if p.is_file()]if b.exists()else[],training_or_Slurm_modified=False,CP_content_not_read=True)))
