from pathlib import Path
import json
root=Path('/mnt/data/shaoshitong/search-g1-prefix5-h10011-runtime-assets-20261018/common8/env/lib/python3.10/site-packages/vllm')
print(json.dumps({rel:(root/rel).read_text()for rel in ['executor/gpu_executor.py','executor/multiproc_gpu_executor.py','engine/arg_utils.py','entrypoints/llm.py']}))
