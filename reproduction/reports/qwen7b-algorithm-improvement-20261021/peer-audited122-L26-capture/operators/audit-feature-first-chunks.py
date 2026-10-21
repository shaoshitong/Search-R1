import json,os,pathlib,subprocess,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-audited122-L26-replay-v1')
pub=json.loads((base/'source-publication.json').read_text());root=pathlib.Path(pub['code_root']);assert pub['commit']=='57bd9e337d6d2a75d2603b684d28752bfe3bf520'
if not all((base/('shard-%02d'%i)/'capture/chunk-00000/complete.json').exists() for i in range(4)):
 print(json.dumps(dict(status='WAITING_ALL_FOUR_ACTUAL_FIRST_CHUNKS',complete_shards=[i for i in range(4) if (base/('shard-%02d'%i)/'capture/chunk-00000/complete.json').exists()],epoch=time.time())))
else:
 script='''import json,time,os,pathlib
from reproduction.operations.qwen7b_warmup import replay_probe_layers as r
from reproduction.operations.qwen7b_warmup.replay_bank_capture import verify_completed_chunk
from reproduction.operations.qwen7b_warmup.replay_feature_reader import ReplayFeatureReader
from reproduction.operations.qwen7b_warmup.peer_newseed_ingest import immutable
b=pathlib.Path(%r);out=b/'first-four-chunks-independent-audit.json'
if out.exists():
 print(json.dumps(dict(status='EXISTING_ACTUAL_FIRST_CHUNK_AUDIT_PRESERVED',audit=r.fingerprints(out))))
else:
 started=time.monotonic();prep=r.read(b/'preparation.json');look={r.read(v['path'])['shard']:v for v in prep['selections']};items=[]
 for i in range(4):
  ref=look[i];selection=r.read(r.checked_file(ref)['path']);expected=[v['trajectory_id'] for v in selection['trajectories'][:320]];p=b/('shard-%%02d'%%i)/'capture/chunk-00000';done=verify_completed_chunk(p,expected)
  assert done['source_selection']==ref and done['batching']==prep['batching'] and done['actor_updates']==0 and set(done['feature_manifests'])=={'26'}
  manifest=done['feature_manifests']['26'];m=r.read(manifest['path']);assert m['layout']=='lossless_question_base_xor_policy_v1' and m['capture_layer']==26
  reader=ReplayFeatureReader([manifest],26);assert set(reader.records)==set(expected)
  reader.get([expected[0],expected[159],expected[-1]],'cpu');reader.close()
  items.append(dict(shard=i,chunk=r.fingerprints(p/'complete.json'),manifest=manifest,trajectories=320,files=len(m['files']),payload_bytes=sum(v['bytes'] for v in m['files'].values()),all_payload_SHA_fresh=True,reader_three_exact_rows_reconstructed=True))
 value=dict(status='ALL_FOUR_ACTUAL_FIRST_LAYER26_CHUNKS_FRESH_SHA_AND_READER_PASSED',source_publication=r.fingerprints(b/'source-publication.json'),preparation=r.fingerprints(b/'preparation.json'),shards=items,trajectories=1280,actor_updates=0,probe_updates=0,elapsed_seconds=time.monotonic()-started,epoch=time.time())
 immutable(out,value);print(json.dumps(dict(value,audit=r.fingerprints(out))))
'''%str(base)
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 p=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45)
 assert p.returncode==0,p.stderr[-1500:]
 print(p.stdout.strip())
