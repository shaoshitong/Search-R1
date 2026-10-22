"""One authentic G5 per question per epoch, exact 1:1 Mixed class counts.

All169028 questions occur once. Mixed-capable questions are allocated without
replacement to84514 positive slots; every remaining question has a real
homogeneous G5. No hard pool, cross-question labels or repeated question views.
"""
import copy
import math
import random
from .g10_grouping import _validate,_partitions,_hash
from .balanced_hard_g10_sampler import ROLES,_tuples


class UniqueQuestionG5Sampler:
 def __init__(self,groups,seed=1729,source_cursor=2577,epoch_end_cursors=(3898,5219),rng_state=None,forbidden_qids=(),forbidden_hashes=()):
  blocked,hashes=set(forbidden_qids),set(forbidden_hashes)
  if any(g['qid']in blocked or g.get('question_hash')in hashes for g in groups):raise ValueError('Evaluation identity cannot enter unique-question training')
  own=_validate(groups,True);self.groups={g['qid']:g for g in own};self.all_qids=[g['qid']for g in own]
  self.mixed_eligible=[g['qid']for g in own if 0<sum(g['labels'])<10]
  self.batch_questions=128;self.half=len(own)//2;self.source_cursor=source_cursor;self.boundaries=tuple(epoch_end_cursors);self.sweep_steps=math.ceil(len(own)/128)
  if len(own)!=169028 or len(own)%2 or len(self.mixed_eligible)<self.half or self.boundaries!=tuple(source_cursor+self.sweep_steps*(i+1)for i in range(len(self.boundaries))):raise ValueError('Exact all-question balanced sweep or cursor boundary impossible')
  self.binding=_hash(dict(version='unique_question_one_real_G5_no_hard_v1',groups=own,seed=seed,source_cursor=source_cursor,epoch_end_cursors=self.boundaries,batch_questions=128,forbidden_qids=sorted(blocked),forbidden_hashes=sorted(hashes),rng_inherited=rng_state is not None))
  self.pool_qids_sha256=_hash(self.all_qids);self.rng=random.Random(seed)
  if rng_state is not None:self.rng.setstate(_tuples(rng_state))
  self.epoch_seed_salt=seed;self.accepted_updates=source_cursor;self.epoch=1;self.pending=None;self._partition_cache={};self.coverage_receipts=[];self._begin_epoch()

 @property
 def steps_per_epoch(self):
  return next((x for x in self.boundaries if x>self.accepted_updates),self.boundaries[-1]+1)

 def epoch_at(self,cursor):return 1+sum(cursor>=x for x in self.boundaries)

 def _begin_epoch(self):
  # Preserve E1 sampler entropy while explicitly applying a new epoch seed.
  entropy=self.rng.getrandbits(256)
  self.rng.seed(_hash(dict(source_entropy=entropy,sampling_seed=self.epoch_seed_salt,epoch=self.epoch)))
  mixed=self.rng.sample(self.mixed_eligible,self.half);positive=set(mixed)
  self.orders={'mixed':mixed,'nonmixed':[q for q in self.all_qids if q not in positive]}
  for values in self.orders.values():self.rng.shuffle(values)
  self.positions={r:0 for r in ROLES};self.cycles={r:self.epoch-1 for r in ROLES}

 def next_batch(self):
  if self.pending is not None:raise ValueError('Previous batch must finish its accepted update first')
  if self.accepted_updates>=self.boundaries[-1]:raise ValueError('Requested unique-question training budget already complete')
  before=dict(accepted_updates=self.accepted_updates,epoch=self.epoch,rng_sha256=_hash(self.rng.getstate()),positions=dict(self.positions))
  take=min(64,self.half-self.positions['mixed'])
  if take<=0 or self.positions['mixed']!=self.positions['nonmixed']:raise ValueError('Balanced unique-question position differs')
  qids=[];roles=[];views=[];per_question=[]
  for role in ROLES:
   start=self.positions[role];selected=self.orders[role][start:start+take];self.positions[role]+=take
   for qid in selected:
    g=self.groups[qid];pattern=tuple(g['labels'])
    if pattern not in self._partition_cache:self._partition_cache[pattern]=_partitions(g)
    pool=self._partition_cache[pattern][0 if role=='mixed'else 1]
    indices=self.rng.choice(pool);labels=[g['labels'][i]for i in indices];k=sum(labels)
    if (0<k<5)!=(role=='mixed'):raise ValueError('Selected own G5 strictEM disagrees with its role')
    qids.append(qid);roles.append(role)
    views.append(dict(qid=qid,source_indices=list(indices),trajectory_ids=[g['trajectory_ids'][i]for i in indices],labels=labels,K=k,true_mixed=0<k<5,weight_numerator=1,weight_denominator=1,weight=1.,source_group_size=10))
    per_question.append(dict(qid=qid,role=role,views=1,unique_subsets=1,hard_pool_selected=False,total_weight_numerator=1,total_weight_denominator=1))
  summary=dict(mode='training_unique_question_one_real_G5_no_hard',questions=2*take,covered_questions=2*take,views=2*take,mixed_views=take,nonmixed_views=take,count_balanced=True,weighted_class_mass_balanced=True,weighted_mixed_fraction=.5,per_question_total_weight_exactly_one=True,views_per_question=1,synthetic_labels=False,labels_recomputed_from_own_subset=True,training_only=True,hard_pool_used=False,epoch=self.epoch,source_question_pool_size=len(self.groups),claim_every_question_once_per_epoch=True,tail_batch=2*take<128,selected_views_sha256=_hash(views))
  out=dict(source_qids=qids,roles=roles,grouping=dict(groups=views,per_question=per_question,summary=summary),before=before,token=_hash(dict(cursor=self.accepted_updates,qids=qids,views=views)))
  self.pending=copy.deepcopy(out);return out

 def observe(self,batch,per_view_mixed_BCE):
  values=list(per_view_mixed_BCE)
  if self.pending is None or batch!=self.pending or len(values)!=len(batch['source_qids'])or any(not math.isfinite(float(x))or float(x)<0 for x in values):raise ValueError('Accepted own one-G5 update/loss binding differs')
  self.accepted_updates+=1;self.pending=None
  if self.positions['mixed']==self.half:
   seen=sorted(self.orders['mixed']+self.orders['nonmixed'])
   if seen!=self.all_qids or self.positions['nonmixed']!=self.half or self.accepted_updates not in self.boundaries:raise ValueError('Epoch did not cover every question exactly once')
   self.coverage_receipts.append(dict(epoch=self.epoch,end_cursor=self.accepted_updates,questions=len(seen),unique_questions=len(set(seen)),mixed_views=self.half,nonmixed_views=self.half,views=len(seen),qid_sha256=_hash(seen),expected_qid_sha256=self.pool_qids_sha256,one_real_G5_per_question=True,hard_pool_used=False,tail_batch_questions=68))
   self.epoch+=1;self._begin_epoch()

 def _seen_summary(self):
  seen=sorted(q for r in ROLES for q in self.orders[r][:self.positions[r]])
  return dict(count=len(seen),unique_count=len(set(seen)),qid_sha256=_hash(seen),mixed_count=self.positions['mixed'],nonmixed_count=self.positions['nonmixed'])

 def state_dict(self):
  if self.pending is not None:raise ValueError('Durable sampler state requires completed accepted update')
  return copy.deepcopy(dict(binding=self.binding,rng_state=self.rng.getstate(),orders=self.orders,positions=self.positions,cycles=self.cycles,accepted_updates=self.accepted_updates,epoch=self.epoch,seen_current_epoch=self._seen_summary(),epoch_coverage_receipts=self.coverage_receipts))

 def load_state_dict(self,state):
  if state['binding']!=self.binding or set(state)!=set(self.state_dict()):raise ValueError('Unique-question sampler strategy/source binding differs')
  c=state['accepted_updates'];completed=(c-self.source_cursor)//self.sweep_steps;within=(c-self.source_cursor)%self.sweep_steps
  if type(c)is not int or not self.source_cursor<=c<=self.boundaries[-1]or state['epoch']!=1+completed:raise ValueError('Unique-question epoch/cursor differs')
  if len(state['epoch_coverage_receipts'])!=completed:raise ValueError('Completed epoch coverage receipts absent')
  for i,receipt in enumerate(state['epoch_coverage_receipts']):
   if(receipt['epoch']!=i+1 or receipt['end_cursor']!=self.boundaries[i]or receipt['questions']!=len(self.groups)or receipt['unique_questions']!=len(self.groups)or receipt['mixed_views']!=self.half or receipt['nonmixed_views']!=self.half or receipt['qid_sha256']!=self.pool_qids_sha256 or receipt['expected_qid_sha256']!=self.pool_qids_sha256 or receipt['one_real_G5_per_question']is not True or receipt['hard_pool_used']is not False):raise ValueError('Completed full-epoch coverage proof differs')
  if any(len(state['orders'][r])!=self.half or state['positions'][r]!=within*64 or state['cycles'][r]!=completed for r in ROLES):raise ValueError('Unique balanced role positions differ')
  if sorted(state['orders']['mixed']+state['orders']['nonmixed'])!=self.all_qids or not set(state['orders']['mixed'])<=set(self.mixed_eligible):raise ValueError('Every question must occur exactly once in the two disjoint role orders')
  self.rng.setstate(_tuples(state['rng_state']))
  for key in('orders','positions','cycles','accepted_updates','epoch'):setattr(self,key,copy.deepcopy(state[key]))
  self.coverage_receipts=copy.deepcopy(state['epoch_coverage_receipts']);self.pending=None
  if self._seen_summary()!=state['seen_current_epoch']:raise ValueError('Actual consumed question prefix differs from saved coverage')
