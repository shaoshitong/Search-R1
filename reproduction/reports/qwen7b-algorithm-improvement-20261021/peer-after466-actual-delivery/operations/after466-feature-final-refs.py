import pathlib,json,hashlib,time,subprocess
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
out={}
for n,q in [('243',31010),('262',33536)]:
 b=A/('peer-joint124194-audited'+n+'-L26-v1');p=b/'bank/complete.json'
 if not p.exists():out[n]=dict(status='WAITING_FOR_ACTUAL_FOUR_SHARD_FEATURE_BANK');continue
 v=json.loads(p.read_text());prep=json.loads((b/'preparation.json').read_text());pub=json.loads((b/'source-publication.json').read_text());c=json.loads((b/'cohort-config.json').read_text());assert(v['questions'],v['trajectories'])==(q,q*10)and v['all_own_IDs_strictEM_and_metadata_SHA_verified']and v['actor_updates']==0 and c['raw_complete']==v['raw_complete']and prep['config']==fp(b/'cohort-config.json')and prep['source_commit']==pub['commit']
 out[n]=dict(status='ACTUAL_FROZEN_SUBSET_L26_FEATURE_BANK_ACCEPTED',complete=fp(p),groups=v['outputs']['groups.json'],features_layer26=v['outputs']['features-layer26.json'],sources=v['outputs']['sources.json'],source_publication=fp(b/'source-publication.json'),source_commit=pub['commit'],source_code_root=pub['code_root'],configuration=fp(b/'cohort-config.json'),preparation=fp(b/'preparation.json'),identity_plan=v['identity_plan'],raw_complete=v['raw_complete'],first_chunk_independent_audit=v['first_chunk_independent_audit'],questions=q,trajectories=q*10,domains=v['domains'],manifests=v['manifests'],packed_bytes=v['packed_bytes'],producer_whole_payload_SHA_and_readers_inherited=v['producer_full_payload_SHA_and_reader_qualifications_preserved'],fresh_all_metadata_index_SHA_and_own_IDs=v['all_own_IDs_strictEM_and_metadata_SHA_verified'])
print(json.dumps(dict(epoch=time.time(),records=out)))
