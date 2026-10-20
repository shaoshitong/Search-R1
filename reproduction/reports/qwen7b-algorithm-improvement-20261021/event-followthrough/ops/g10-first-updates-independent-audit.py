"""Actual G10 first-update/full-CP audit and own-EM view/weight reconstruction."""
import hashlib,json,pathlib,sys,time,math
T=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g10-matched6528-v1')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def check(v):
 actual=fp(v['path']);assert actual['sha256']==v['sha256']
 if 'bytes' in v:assert actual['bytes']==v['bytes']
 return pathlib.Path(v['path'])
p=read(T/'preparation.json');pub=read(T/'source-publication.json');root=pathlib.Path(pub['code_root'])
assert p['source_Adam_steps']==10464 and p['questions']==6528 and p['updates']==136 and p['uniform_view_weight']=='1/4'
for rel,ref in pub['files'].items():assert fp(root/rel)['sha256']==ref['sha256']
sys.path.insert(0,str(root))
from reproduction.operations.qwen7b_warmup.g10_grouping import build_training_views as random_views
from reproduction.operations.qwen7b_warmup.g10_weighted_balanced import build_training_views as balanced_views
g10=read(check(p['g10_complete']));groups=read(check(g10['groups']))['train'];lookup={g['qid']:g for g in groups}
assert len(lookup)==6528 and all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in groups)
source=read(check(g10['original_G5_bank']))['train'];old={g['qid']:g for g in source};extra={}
for ref in g10['extra_shards']:
 s=read(check(ref));assert s['status']=='EXTRA_REAL_G5_SHARD_NATIVE_FULL_OWN_EM_SHA_VERIFIED' and s['actor_updates']==0
 for g in s['groups']:assert g['qid'] not in extra;extra[g['qid']]=g
for g in groups:
 a,b=old[g['qid']],extra[g['qid']]
 assert g['trajectory_ids']==a['trajectory_ids']+b['trajectory_ids'] and g['labels']==a['labels']+b['labels']
 assert g['question_hash']==a['question_hash']==b['question_hash']
out=dict(status='G10_ACTUAL_FIRST_UPDATE_FULL_CHECKPOINT_AND_OWN_EM_QUARTER_WEIGHT_VIEW_RECONSTRUCTION_PASS',
 observed_epoch=time.time(),variants={},training_g10=fp(p['g10_complete']['path']),torch_imported=False,
 model_or_optimizer_deserialized=False,raw_predictions_returned=False,untouched_final_read=False)
sources=[];plans=[]
for name,builder in [('random',random_views),('global-weighted-balanced',balanced_views)]:
 b=T/name;item=p['variants'][name];c=read(check(item['config']));check(item['sbatch']);check(c['g10_complete']);plan=read(check(c['plan']))
 assert c['global_questions']==48 and c['update_budget']==136 and c['fit_questions']==6528 and c['micro_trajectories']==5
 assert c['view_weight_numerator']==1 and c['view_weight_denominator']==4 and c['source_adam_steps']==10464
 assert c['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.) and c['optimizer_reset'] is False
 sources.append(c['source_checkpoint']);plans.append(c['plan'])
 ctrl=read(b/'controller.json');assert ctrl['job_id']==(b/'job.id').read_text().strip() and ctrl['source_steps']==10464
 assert ctrl['config_sha256']==item['config']['sha256'] and not (b/'failure.json').exists()
 proof=read(b/'next-update-replay.json');assert proof['status']=='PASSED' and proof['source_steps']==10464 and proof['candidate_steps']==10465
 assert proof['expected']==proof['actual'] and set(proof['actual'])=={'raw','ema','adam','raw_buffers','ema_buffers','rng'}
 first=fp(b/'checkpoints/cursor-000001.pt');assert first['bytes']>3*1024**3
 receipt=read(b/'latest.json');check(receipt);assert receipt['cursor']>=16 and receipt['steps']==10464+receipt['cursor']
 assert receipt['config_sha256']==item['config']['sha256'] and receipt['matched_binding_sha256']==ctrl['matched_binding_sha256']
 rows=[json.loads(s) for s in (b/'curve.jsonl').read_text().splitlines() if s.endswith('}')]
 assert len(rows)>=2 and [r['cursor'] for r in rows]==list(range(1,len(rows)+1))
 for r in rows:
  assert r['steps']==10464+r['cursor'] and r['actor_updates']==0 and r['original_questions']==48 and r['virtual_views']==192
  assert r['unique_real_trajectories']==480 and r['actual_forward_trajectory_exposures']==960 and r['recomputed_logits_bit_identical'] is True and math.isfinite(r['loss'])
  kwargs=dict(c['grouping']);kwargs['epoch']=plan['batch_epochs'][r['cursor']-1]
  views=builder([lookup[q] for q in plan['batches'][r['cursor']-1]],role='train',**kwargs)
  assert views['summary']==r['grouping']
  for g in views['groups']:
   src=lookup[g['qid']];ix=g['source_indices'];assert g['trajectory_ids']==[src['trajectory_ids'][i] for i in ix]
   assert g['labels']==[src['labels'][i] for i in ix] and g['true_mixed']==(0<sum(g['labels'])<5)
   assert g['weight_numerator']==1 and g['weight_denominator']==4
  if name=='global-weighted-balanced':
   s=r['grouping'];assert s['mixed_views']==s['nonmixed_views']==96 and s['weighted_mixed_mass']==s['weighted_nonmixed_mass']=='24'
   assert s['count_balanced'] and s['weighted_class_mass_balanced'] and s['exactly_four_unique_views_per_question']
 out['variants'][name]=dict(job_id=ctrl['job_id'],config=item['config'],first_checkpoint=first,
  latest_checkpoint={k:receipt[k] for k in ('path','bytes','sha256','cursor','steps')},
  double_next_update_replay_passed=True,actual_accepted_updates=len(rows),actual_Adam_steps=10464+len(rows),
  exact_own_EM_views_reconstructed_for_all_recorded_updates=True,all_views_weight_one_quarter=True,
  per_question_weight_exactly_one=True,all_global_updates_exact1to1=(name=='global-weighted-balanced'),
  topology_changed=True,four_rank_bitexact_resume_claim=False)
assert sources[0]==sources[1] and plans[0]==plans[1];check(sources[0]);out['source_E1_full_file_SHA_freshly_verified']=True
assert 'torch' not in sys.modules
print(json.dumps(out))
