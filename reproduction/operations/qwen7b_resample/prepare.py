"""Reuse original question/token inputs, replace only run/seed and partition."""
import argparse,hashlib,json,uuid,time
from pathlib import Path
from reproduction.operations.qwen7b_warmup.download_model import atomic

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--assets',required=True);p.add_argument('--retrieval-config',required=True);a=p.parse_args();b=Path(a.base);assets=Path(a.assets);b.mkdir(parents=True,exist_ok=True)
 if(b/'config.json').exists():print('EXISTING_CONFIG_NO_REPARTITION');return
 original=json.loads((assets/'imported/source/config.json').read_text());assert original['seed']==1729
 rows=[]
 for file in sorted((assets/'imported/source/inputs').glob('shard-*/batch-*.json')):
  v=json.loads(file.read_text())
  if v['split']=='train':rows.extend(v['rows'])
 rows.sort(key=lambda x:x['source_ordinal']);assert len(rows)==169028 and len({x['qid']for x in rows})==169028
 assert all(x['split']=='train'for x in rows)
 r=json.loads(Path(a.retrieval_config).read_text());c=dict(original,base=str(b),model_dir=str(assets/'official-qwen2.5-7b-instruct-a09a35458c70'),seed=20261021,run_id=uuid.uuid4().hex,sampling_attempt_id='independent-seed20261021-g5-20261020',train_questions=169028,train_trajectories=845140,diagnostic_questions=0,diagnostic_trajectories=0,sampling_shards=2,warmup_only=True,actor_hidden_native=False,actor_optimizer_updates=0,search_url='http://127.0.0.1:28831/retrieve',source_seed=original['seed'],source_run_id=original['run_id'],source_input_root=str(assets/'imported/source/inputs'),hidden_cached=False,source_commit=r['new_source_commit'])
 for k in ['retrieval_index','retrieval_corpus','retrieval_encoder','retrieval_data_cache']:c[k]=r[k]
 manifest=[]
 for shard in range(2):
  target=b/'inputs'/('shard-%02d'%shard);target.mkdir(parents=True,exist_ok=True);own=rows[shard::2]
  for j,start in enumerate(range(0,len(own),128)):
   path=target/('batch-%06d.json'%j);v=dict(shard=shard,split='train',rows=own[start:start+128],model_revision=c['model_revision']);atomic(path,v);manifest.append(dict(path=str(path),questions=len(v['rows']),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 atomic(b/'inputs/manifest.json',dict(batches=manifest,questions=len(rows),trajectories=845140,ordered_qid_sha256=hashlib.sha256(json.dumps([x['qid']for x in rows]).encode()).hexdigest(),seed=c['seed'],diagnostic_not_sampled=True));atomic(b/'config.json',c);print(json.dumps(dict(run_id=c['run_id'],questions=len(rows),trajectories=845140,seed=c['seed'],batches=len(manifest))))
if __name__=='__main__':main()
