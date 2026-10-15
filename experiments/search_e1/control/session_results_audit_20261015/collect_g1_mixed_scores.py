from pathlib import Path
import sys,json
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'g1-paper-comparison-20261014'));from h100_remote import call
c=r'''from pathlib import Path
import json,hashlib,math,datetime,collections
r=Path('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v2');j=r/'journal';p=j/'final-result.json';o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'result_exists':p.exists()}
def canonical(d):return json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(d):return hashlib.sha256(canonical(d)).hexdigest()
binding=json.loads((j/'binding.json').read_text());o['binding']=binding;o['binding_sha256']=sha(binding)
rows=[];previous=None;proof=[]
for b in sorted(j.glob('batch-????????-????????.json')):
 raw=b.read_bytes();v=json.loads(raw);assert v['start']==len(rows);assert v['stop']-v['start']==len(v['rows']);assert v['previous_batch_sha256']==previous;assert v['binding_sha256']==sha(binding);assert v['rows_sha256']==sha(v['rows']);rows.extend(v['rows']);previous=sha(v);proof.append([b.name,hashlib.sha256(raw).hexdigest()])
o['complete_batches']=len(proof);o['durable_rows']=len(rows);o['batch_chain_verified']=True;o['batch_hash_inventory_sha256']=sha(proof)
assert len(rows)==51713 and [x['ordinal'] for x in rows]==list(range(51713));assert all(x['em'] in (0,1) and 0<=x['f1']<=1 for x in rows)
occ=[[i,v['question_id'],v['data_source']] for i,v in enumerate(rows)];assert sha(occ)==binding['occurrence_sha256'];o['occurrence_binding_verified']=True
o['em']=math.fsum(x['em'] for x in rows)/len(rows);o['f1']=math.fsum(x['f1'] for x in rows)/len(rows);o['correct']=sum(x['em'] for x in rows);o['counts']=dict(collections.Counter(x['data_source'] for x in rows));o['source_metrics']={s:{'count':o['counts'][s],'em':math.fsum(x['em'] for x in rows if x['data_source']==s)/o['counts'][s],'f1':math.fsum(x['f1'] for x in rows if x['data_source']==s)/o['counts'][s]} for s in o['counts']}
if p.exists():
 raw=p.read_bytes();d=json.loads(raw);o['result_file']={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime':p.stat().st_mtime};assert d['rows']==rows;assert d['weighted_em']==o['em'] and d['weighted_f1']==o['f1'];assert d['correct']==o['correct'];o['published_final_matches_journal']=True
print(json.dumps(o))'''
v=call('h100-8',c);p=D/('g1-mixed-final-scores-'+sys.argv[1]+'.json');p.write_text(json.dumps(v,ensure_ascii=False,indent=2));print(json.dumps(v,ensure_ascii=False))
