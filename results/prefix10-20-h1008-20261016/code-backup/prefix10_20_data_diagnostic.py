import pathlib,json,datetime,sys
repo=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix10-20-h1008-20261016');sys.path.insert(0,str(repo))
from reproduction.h1008_qprobe.management_transport import remote
work=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work')
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
script=r'''
import pathlib,json,hashlib,datetime,collections,itertools,os
import numpy as np
base=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix10-20-derived-inputs-20261016-ea7ddae805a94bd4a7cfb55e484fdb46');source=base/'inputs'
rows=[json.loads(x) for x in (source/'index.jsonl').read_text().splitlines()]
ids=np.load(source/'packed_input_ids.npy',allow_pickle=False,mmap_mode='r');offsets=np.load(source/'offsets.npy',allow_pickle=False,mmap_mode='r')
proof=json.loads((source/'scan-complete.json').read_text())
result=dict(role='prefix10_20_existing_inputs_scalar_diagnostics',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_scan_sha256=hashlib.sha256((source/'scan-complete.json').read_bytes()).hexdigest(),input_count=len(rows),global_max_tokens=proof['global_max_tokens'],no_new_inference=True,no_raw_tokens_exported=True,no_repeated_full_sha=True,scope='Existing fixed5120question training and fixednew512question diagnosis, five sibling trajectories per question; identities not exported',splits={})
for split in ('train','test'):
 chosen=[r for r in rows if r['split']==split];groups=collections.defaultdict(list)
 for r in chosen:groups[r['qid']].append(r)
 assert len(chosen)==(25600 if split=='train' else 2560) and all(len(v)==5 for v in groups.values())
 em=sum(r['strict_em'] for r in chosen)
 item=dict(questions=len(groups),trajectories=len(chosen),strict_em_correct=em,strict_em_incorrect=len(chosen)-em,positive_fraction=em/len(chosen),majority_accuracy=max(em,len(chosen)-em)/len(chosen),prefix={})
 for length in (10,20):
  sequences={};lens=collections.Counter();eos=0
  for r in chosen:
   start=int(offsets[r['row']])+r['prompt_length'];k=min(length,r['prefix_length']);seq=tuple(int(x) for x in ids[start:start+k]);sequences[r['trajectory_id']]=seq;lens[k]+=1
   eos+=r['eos_token_id'] in seq
  diversity=collections.Counter();equal=0;all_same=0;position_equal=[0]*length;position_pairs=[0]*length
  for values in groups.values():
   seqs=[sequences[r['trajectory_id']] for r in values];distinct=len(set(seqs));diversity[distinct]+=1;all_same+=distinct==1
   for a,b in itertools.combinations(seqs,2):
    equal+=a==b
    for j in range(min(len(a),len(b),length)):position_pairs[j]+=1;position_equal[j]+=a[j]==b[j]
  pairs=len(groups)*10
  item['prefix'][str(length)]=dict(length_distribution={str(k):v for k,v in sorted(lens.items())},short_count=sum(v for k,v in lens.items() if k<length),short_fraction=sum(v for k,v in lens.items() if k<length)/len(chosen),observed_eos_count=eos,observed_eos_fraction=eos/len(chosen),distinct_sequences_per_question_distribution={str(k):v for k,v in sorted(diversity.items())},all_five_identical_questions=all_same,all_five_identical_fraction=all_same/len(groups),identical_sibling_pairs=equal,sibling_pairs=pairs,pair_identical_fraction=equal/pairs,pair_position_agreement=[dict(position=j+1,compared_pairs=position_pairs[j],identical_pairs=position_equal[j],fraction=position_equal[j]/position_pairs[j] if position_pairs[j] else None) for j in range(length)])
 result['splits'][split]=item
out=base/('data-diagnostics-__TAG__.json')
with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(summary=result,remote_summary_path=str(out),remote_summary_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
'''.replace('__TAG__',tag)
socket=json.loads((work/'qprobe3-management-current.json').read_text())['socket']
res=json.loads(remote(script,work/('prefix10-20-data-diagnostic-transport-'+tag+'.json'),control_path=socket))
p=work/('prefix10-20-data-diagnostic-summary-'+tag+'.json');p.write_text(json.dumps(res,indent=2,allow_nan=False))
print(json.dumps(dict(local_summary=str(p),local_summary_sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest(),remote_summary_path=res['remote_summary_path'],remote_summary_sha256=res['remote_summary_sha256'],splits=res['summary']['splits'])))
