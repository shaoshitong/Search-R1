# Read-only bounded pilot verification; aggregate identity/own-EM results only.
import pathlib,json,gzip,hashlib,time,collections
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
b=B/'algorithm-improvement-20261021/training-extraG5-128-20261021/materialized'
def read(p):return json.loads(pathlib.Path(p).read_text())
def checked(item):
 p=pathlib.Path(item['path']);assert str(p.resolve()).startswith(str(b.resolve())+'/')
 assert p.stat().st_size==item['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
 assert h.hexdigest()==item['sha256']
 return p
def rows(p):
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(p,'rt') as f:
  for line in f:yield json.loads(line)
c=read(b/'config.json');assert c['source_commit'].startswith('c8b02d3b22b6') and c['extra_response_indices']==list(range(5,10))
qplan=read(b/'inputs/questions.json');original={r['qid']:set(r['trajectory_ids']) for r in qplan['groups']}
result=dict(observed_epoch=time.time(),job_id='742987',source_commit=c['source_commit'],actor_updates_authorized=c['actor_updates_authorized'],verified_batches=[])
for i in range(4):
 r=b/'sampling'/('shard-%02d'%i)
 complete=sorted((r/'batches').glob('*/complete.json')) if (r/'batches').exists() else []
 if not complete:continue
 # First sealed batch only per shard. Never dump text, IDs or token arrays.
 p=complete[0];batch=read(p)
 assert batch['run_id']==c['run_id'] and batch['actor_optimizer_updates']==0
 expected=read(b/'inputs'/('shard-%02d'%i)/(batch['batch']+'.json'))['rows']
 expected_qids={row['qid'] for row in expected}
 prefix=batch['prefix'];full=batch['full']
 assert prefix['sampling_attempt_id']==full['sampling_attempt_id']==c['sampling_attempt_id']
 pp=checked(prefix['prefix']);fp=checked(full['full']);ep=checked(full['own_em'])
 prefixes={};fulls={};labels={}
 for row in rows(pp):
  tid=row['trajectory_id'];assert tid not in prefixes;prefixes[tid]=row
 for row in rows(fp):
  tid=row['trajectory_id'];assert tid not in fulls;fulls[tid]=row
 for row in rows(ep):
  tid=row['trajectory_id'];assert tid not in labels and type(row['strict_em']) is int and row['strict_em'] in (0,1);labels[tid]=row
 assert set(prefixes)==set(fulls)==set(labels)
 indices=collections.Counter();bits=collections.Counter();groups={};journal_chunks=0
 for tid,pre in prefixes.items():
  f=fulls[tid];em=labels[tid];qid=pre['qid'];index=pre['response_index']
  assert qid in expected_qids and qid in original and tid not in original[qid]
  assert index in range(5,10) and f['response_index']==index
  assert qid==f['qid']==em['qid'] and pre['split']==f['split']=='train'
  assert pre['prefix_sha256']==f['prefix_sha256']==em['prefix_sha256']
  assert type(f['strict_em']) is int and f['strict_em']==em['strict_em']
  assert f['sampling_journal'] and f['behavior_turns']
  journal_chunks+=len(f['sampling_journal']);indices[index]+=1;bits[f['strict_em']]+=1
  groups.setdefault(qid,[]).append((index,f['strict_em']))
 assert set(groups)==expected_qids
 assert all(sorted(index for index,label in own)==list(range(5,10)) for own in groups.values())
 sums=collections.Counter(sum(label for index,label in own) for own in groups.values())
 result['verified_batches'].append(dict(shard=i,batch=batch['batch'],questions=len(groups),trajectories=len(prefixes),response_index_counts=dict(indices),own_EM_counts=dict(bits),group_correct_count_histogram=dict(sums),mixed_questions=sum(n for k,n in sums.items() if 0<k<5),actual_sampling_journal_chunks=journal_chunks,prefix_full_own_EM_SHA_verified=True,all_actual_ids_disjoint_from_original_G5=True,actor_optimizer_updates=0))
print(json.dumps(result))
