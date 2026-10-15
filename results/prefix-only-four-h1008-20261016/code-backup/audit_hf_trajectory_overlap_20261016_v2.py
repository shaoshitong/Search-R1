"""CPU-only HF bytes-in-memory comparison against existing remote identity index."""
import importlib.util
import json
import pathlib
import shlex
import subprocess
import hashlib
import datetime

ROOT=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/datasets/Trajectory_in_Search/official_qwen25_frozen_g5_20261016')
WORK=pathlib.Path(__file__).parent
SOURCE='/data/shaoshitong/Search-G1-prefix-only-four-inputs-20261016-96a20179b7784025b647c6aa0279303e/inputs'
RUNTIME='/data/shaoshitong/Search-G1-a800-fresh200-16034e59031e4f21a323ebb2556bef7f/repo'
spec=importlib.util.spec_from_file_location('existing_a800_transport','/Users/shaoshitong/Documents/Search-G1/qprobe-qwen-native-selection-metrics-fix/reproduction/a800_common/deploy.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
remote=r'''
import sys,io,json,hashlib,pathlib,datetime,unicodedata
import pyarrow.parquet as pq
root=pathlib.Path(SOURCE)
proof=json.loads((root/'scan-complete.json').read_text())
indexraw=(root/'index.jsonl').read_bytes();assert hashlib.sha256(indexraw).hexdigest()==proof['files']['index.jsonl']['sha256']
rows=[json.loads(x)for x in indexraw.decode().splitlines()]
original={split:[r for r in rows if r['split']==split]for split in ('train','test')}
header=json.loads(sys.stdin.buffer.readline());out={}
# Only canonical question metadata columns, never rollout/model tensors.
import os
sys.path.insert(0, RUNTIME)
from search_g1.qprobe_features import original_question_and_id
source_binding=json.loads((root.parent/'sources.json').read_text())
canonical=pathlib.Path(source_binding['canonical_train']['path'])
target_qids={str(r['qid'])for r in rows}
source_content={}
for batch in pq.ParquetFile(canonical).iter_batches(batch_size=1024,columns=['prompt','extra_info','data_source']):
 for row in batch.to_pylist():
  text,qid=original_question_and_id(row)
  if qid in target_qids:source_content[qid]=text
assert set(source_content)==target_qids
def norm(q):return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',q).split()).encode()).hexdigest()
def select(names,options):
 result=[name for name in options if name in names]
 return result[0]if result else None
for item in header:
 data=sys.stdin.buffer.read(item['bytes']);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
 pf=pq.ParquetFile(io.BytesIO(data));names=pf.schema_arrow.names
 keys={'qid':select(names,('qid','question_id','original_question_id')),'trajectory_id':select(names,('trajectory_id','behavior_trajectory_id','response_id')),'question':select(names,('question','original_question','question_text')),'question_hash':select(names,('question_hash','content_sha256')),'strict_em':select(names,('final_binary_em','strict_em','correct','em','label'))}
 values=pf.read(columns=list(dict.fromkeys(v for v in keys.values()if v))).to_pylist();split=item['split'];ref=original[split]
 record=dict(row_count=pf.metadata.num_rows,columns=names,identity_columns=keys,source_rows=len(ref),local_parquet_sha256=item['sha256'])
 refq={str(r['qid'])for r in ref};reft={str(r['trajectory_id'])for r in ref};refhash={r['question_hash']for r in ref};refnorm={norm(source_content[str(r['qid'])])for r in ref}
 if keys['qid']:
  qids={str(r[keys['qid']])for r in values};record.update(unique_questions=len(qids),qid_overlap=len(qids&refq),qid_only_hf=len(qids-refq),qid_only_source=len(refq-qids),qid_set_equal=qids==refq)
 if keys['trajectory_id']:
  tids={str(r[keys['trajectory_id']])for r in values};record.update(unique_trajectories=len(tids),trajectory_id_overlap=len(tids&reft),trajectory_id_set_equal=tids==reft)
 if keys['question']:
  hashes={norm(r[keys['question']])for r in values};record.update(normalized_question_content_overlap=len(hashes&refnorm),normalized_question_unique=len(hashes),normalized_question_content_set_equal=hashes==refnorm)
 if keys['question_hash']:
  hs={r[keys['question_hash']]for r in values};record.update(question_hash_overlap=len(hs&refhash),question_hash_set_equal=hs==refhash)
 if keys['trajectory_id'] and keys['strict_em']:
  expected={str(r['trajectory_id']):r['strict_em']for r in ref};record.update(strict_em_disagreements=sum(int(r[keys['strict_em']])!=expected.get(str(r[keys['trajectory_id']]))for r in values),strict_em_correct=sum(int(r[keys['strict_em']])for r in values))
 cross=original['test'if split=='train'else'train'];crossq={str(r['qid'])for r in cross};crosshash={norm(source_content[str(r['qid'])])for r in cross};crosst={str(r['trajectory_id'])for r in cross}
 record['cross_other_split_overlap']=dict(qid=len(qids&crossq)if keys['qid']else None,trajectory_id=len(tids&crosst)if keys['trajectory_id']else None,normalized_content=len(hashes&crosshash)if keys['question']else None)
 out[split]=record
print(json.dumps(dict(role='HF_actual_parquet_vs_existing_teacher_and_new512_identity_audit',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_index=str(root/'index.jsonl'),source_index_sha256=hashlib.sha256(indexraw).hexdigest(),source_scan_sha256=hashlib.sha256((root/'scan-complete.json').read_bytes()).hexdigest(),canonical_question_metadata_path=str(canonical),canonical_question_metadata_count=len(source_content),normalization='NFKC_then_whitespace_collapse_then_SHA256_UTF8',splits=out,GPU_used=False,remote_files_written=False,training_state_modified=False,original_rollouts_or_tensors_downloaded=False)),flush=True)
'''.replace('SOURCE',repr(SOURCE)).replace('RUNTIME',repr(RUNTIME))
header=[];parts=[]
for split in ('train','test'):
 path=ROOT/(split+'-00000-of-00001.parquet');data=path.read_bytes();header.append(dict(split=split,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));parts.append(data)
payload=json.dumps(header).encode()+b'\n'+b''.join(parts)
wrapper="import os; env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2'); os.execve('/data/search-g1/env/bin/python',['/data/search-g1/env/bin/python','-c',"+repr(remote)+"],env)"
r=subprocess.run(mod.ssh_command(wrapper),input=payload,capture_output=True,timeout=180)
transport=dict(returncode=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),HF_bytes_sent_for_in_memory_CPU_read=len(payload),remote_files_written=False)
p=WORK/'hf-trajectory-overlap-transport-20261016-v2.json'
with p.open('x')as f:json.dump(transport,f,indent=2);f.write('\n')
if r.returncode:print(json.dumps(transport));raise SystemExit(r.returncode)
summary=json.loads(transport['stdout']);summary.update(local_dataset=str(ROOT),repo_id='Sutoonq/Trajectory_in_Search',config='official_qwen25_frozen_g5_20261016',repo_revision=json.loads((ROOT/'DOWNLOAD-MANIFEST.json').read_text())['repo_revision'])
p=ROOT/'OVERLAP-AUDIT-v2.json'
with p.open('x')as f:json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(summary,indent=2));print('evidence_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
