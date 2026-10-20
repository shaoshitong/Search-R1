from pathlib import Path
import json,re,time
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');rows=[];pids={1437188}
for p in sorted((b/'pool/jobs').glob('job-shared5-1437188*/attempt-*')):
 v={'path':str(p)}
 for q in p.glob('*children.json'):
  x=json.loads(q.read_text());v[q.name]=x;pids.update(x.get('own_process_groups',[]));pids.update(x.get('engine_processes',{}).values())
  if x.get('last_reaped_sampler_pid'):pids.add(x['last_reaped_sampler_pid'])
 q=p/'sampler.log'
 if q.exists():pids.update(int(x)for x in re.findall(r'VllmWorkerProcess pid=(\d+)',q.read_text(errors='replace')))
 rows.append(v)
model=Path('/mnt/data/shaoshitong/qwen7b-probe-assets-20261020/official-qwen2.5-7b-instruct-a09a35458c70/config.json');c=json.loads(model.read_text())
print(json.dumps(dict(epoch=time.time(),attempts=rows,observed_own_pids=sorted(pids),model_config={k:c.get(k)for k in ['model_type','hidden_size','num_attention_heads','num_key_value_heads','num_hidden_layers','torch_dtype']})))
