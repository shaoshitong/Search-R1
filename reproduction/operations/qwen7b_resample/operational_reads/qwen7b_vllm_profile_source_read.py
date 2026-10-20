from pathlib import Path
import json
site=Path('/mnt/data/shaoshitong/search-g1-prefix5-h10011-runtime-assets-20261018/common8/env/lib/python3.10/site-packages/vllm')
out={}
for rel,terms,span in [('worker/worker.py',['determine_num_available_blocks','init_gpu_memory','get_cache_block_size_bytes'],65),('engine/arg_utils.py',['worker_cls','num_gpu_blocks_override'],5),('executor/multiproc_gpu_executor.py',['WorkerWrapperBase','worker_module_name','worker_class_name'],12),('worker/worker_base.py',['init_worker','worker_module_name','worker_class_name'],18)]:
 p=site/rel;lines=p.read_text().splitlines();chosen=set()
 for i,line in enumerate(lines):
  if any(term in line for term in terms):chosen.update(range(max(0,i-3),min(len(lines),i+span)))
 out[rel]='\n'.join(str(i+1)+':'+lines[i]for i in sorted(chosen))
print(json.dumps(out))
