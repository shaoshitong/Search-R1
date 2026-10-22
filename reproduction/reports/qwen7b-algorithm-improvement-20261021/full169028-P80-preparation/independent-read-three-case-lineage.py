from pathlib import Path
import json,time,gzip,hashlib
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';T=A/'joint169028-native-training-bank-v1';qids={'["hotpotqa", "hotpotqa:train:70604"]','["hotpotqa", "hotpotqa:train:2094"]','["hotpotqa", "hotpotqa:train:81793"]'}
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
 p=Path(p);r=p.read_bytes();return dict(path=str(p),bytes=len(r),sha256=hashlib.sha256(r).hexdigest())
def selected(p):
 v=read(p);items=v['train']if isinstance(v,dict)and'train'in v else v;return[g for g in items if g['qid']in qids]
out=dict(time=time.time(),canonical_group_ref=ref(T/'groups.json'),canonical_selected=selected(T/'groups.json'),source_banks=[]);complete=read(T/'complete.json')
for item in complete['source_refs']:
 groups=selected(item['groups']['path'])
 if not groups:continue
 done=read(item['complete']['path']);keys=('status','preparation','raw_complete','source_resolution_contract','physical_source_final_readback','source_refs');small={k:done[k]for k in keys if k in done};out['source_banks'].append(dict(key=item['key'],groups_ref=item['groups'],selected=groups,complete_ref=item['complete'],complete_lineage=small))
inputpath=B/'peer-newseed-h100-20261021/audited138-after971-20261021-51057f37/raw/audited138-after971-20261021-51057f37/group-00006/source/inputs/shard-01/batch-000420.json';inp=read(inputpath);out['source_inputs']=[]
for row in inp['rows']:
 if row['qid']in qids:out['source_inputs'].append({k:row[k]for k in('qid','question_hash','candidate_ordinal','reward_model','data_source')if k in row})
out['final4096_read']=False;out['label_or_training_modified']=False;print(json.dumps(out))
