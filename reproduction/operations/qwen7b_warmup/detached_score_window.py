"""Deterministic, graph-free unique-question FIFO for a cached TopK score window.

Caller supplies global current entries in the exact order of its autograd score
vector. prepare() removes prior views of those qids, then returns recent cached
rows followed by current rows. Only current scores are differentiated by runtime.
commit() runs AFTER a possible Adam step but enqueues PRE-update detached scores.
Sampler/RNG and distributed all_gather are owned by the common training runtime.
"""
import hashlib,json,math,numbers

SCHEMA='unique_question_detached_score_window_v1'

def _copy(value):return json.loads(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False))
def _sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def _integer(value,name):
 if isinstance(value,bool)or not isinstance(value,numbers.Integral)or value<0:raise ValueError(name+' must be a nonnegative integer')
 return int(value)
def _no_features(value):
 if isinstance(value,dict):
  if {'features','hidden_features','hidden_states','feature_tensor','activations'}&set(value):raise ValueError('Cache may not contain features or activations')
  for v in value.values():_no_features(v)
 elif isinstance(value,(list,tuple)):
  for v in value:_no_features(v)
def _entry(value):
 _no_features(value)
 # JSON copying rejects Tensor/autograd objects rather than silently detaching.
 row=_copy(value);row['qid']=str(row['qid'])
 ids=row['trajectory_ids'];labels=row['labels']
 if not ids or len(ids)!=len(labels)or len(set(map(str,ids)))!=len(ids):raise ValueError('Distinct own trajectory IDs and their actual strict EM labels are required')
 if any(type(y)not in(int,bool)or int(y)not in(0,1)for y in labels):raise ValueError('Strict EM labels must be binary')
 row['labels']=[int(y)for y in labels];mixed=0<sum(row['labels'])<len(labels)
 if type(row['true_mixed'])is not bool or row['true_mixed']!=mixed:raise ValueError('true_mixed must use exactly this selected view, not full G10')
 score=row['cached_score']
 if isinstance(score,bool)or not isinstance(score,numbers.Real)or not math.isfinite(score):raise ValueError('A finite detached scalar cached_score is required')
 row['cached_score']=float(score);row['scored_step']=_integer(row['scored_step'],'scored_step');row['scored_batch_attempt']=_integer(row['scored_batch_attempt'],'scored_batch_attempt')
 if row['score_model']!='raw':raise ValueError('Cache score_model must be raw')
 return row

def _ages(rows,step):
 ages=[step-r['scored_step']for r in rows]
 if any(a<0 for a in ages):raise ValueError('Cached score version is newer than current raw model')
 return dict(age_units='actual_Adam_updates',age_min=min(ages)if ages else None,age_max=max(ages)if ages else None,age_mean=sum(ages)/len(ages)if ages else None,scored_step_min=min((r['scored_step']for r in rows),default=None),scored_step_max=max((r['scored_step']for r in rows),default=None))

class DetachedScoreWindow:
 def __init__(self,capacity=2018,target_k=500,reference_population=2018):
  self.capacity=_integer(capacity,'capacity');self.target_k=_integer(target_k,'target_k');self.reference_population=_integer(reference_population,'reference_population')
  if not 1<=self.target_k<=self.reference_population:raise ValueError('Require1<=target_k<=reference_population')
  self._rows=[];self._pending=None;self._last_model_version=None;self._committed_batches=0
 @property
 def has_pending(self):return self._pending is not None
 @property
 def ring_sha256(self):return _sha(self._rows)
 def prepare(self,current_entries,model_version):
  if self._pending is not None:raise RuntimeError('Commit the prepared current batch before assembling another window')
  step=_integer(model_version,'model_version');current=[_entry(r)for r in current_entries];B=len(current)
  if B==0:raise ValueError('Current global gradient batch must be nonempty')
  if self.capacity and B>self.capacity:raise ValueError('Current batch exceeds the requested total score-window capacity')
  qids=[r['qid']for r in current]
  if len(set(qids))!=B:raise ValueError('Current global batch must contain unique questions')
  if any(r['scored_step']!=step for r in current)or len({r['scored_batch_attempt']for r in current})!=1:raise ValueError('Current detached pre-update scores must share the actual current model and batch attempt')
  if self._last_model_version is not None and step!=self._last_model_version:raise ValueError('Actual Adam version must continue the saved cache state')
  previous=len(self._rows);own=set(qids);self._rows=[r for r in self._rows if r['qid']not in own]
  room=max(0,self.capacity-B);cached=self._rows[-room:]if room else[];window=cached+current;N=len(window);K=(N*self.target_k+self.reference_population-1)//self.reference_population
  if len({r['qid']for r in window})!=N:raise ValueError('Unique-question FIFO invariant failed')
  diagnostics=dict(N_eff=N,K=K,current_rows=B,cached_rows=len(cached),ring_rows=len(self._rows),removed_previous_current_qids=previous-len(self._rows),duplicate_window_qids=0,model_version=step,reference_population=self.reference_population,ring_sha256=self.ring_sha256,window_sha256=_sha(window),**_ages(cached,step))
  self._pending=dict(current_entries=current,pre_update_model_version=step,scored_batch_attempt=current[0]['scored_batch_attempt'])
  return dict(cached_rows=_copy(cached),current_rows=_copy(current),rows=_copy(window),cached_scores=[r['cached_score']for r in cached],cached_labels=[int(r['true_mixed'])for r in cached],current_labels=[int(r['true_mixed'])for r in current],N_eff=N,K=K,current_offset=len(cached),diagnostics=diagnostics)
 def commit(self,post_update_step):
  if self._pending is None:raise RuntimeError('No prepared batch to enqueue')
  after=_integer(post_update_step,'post_update_step');before=self._pending['pre_update_model_version']
  if after not in(before,before+1):raise ValueError('One global batch can perform zero or one actual Adam update')
  self._rows+=self._pending['current_entries']
  self._rows=self._rows[-self.capacity:]if self.capacity else[]
  self._pending=None;self._last_model_version=after;self._committed_batches+=1
  return dict(ring_rows=len(self._rows),ring_sha256=self.ring_sha256,model_version=after,committed_batches=self._committed_batches,**_ages(self._rows,after))
 def state_dict(self):
  return _copy(dict(schema=SCHEMA,capacity=self.capacity,target_k=self.target_k,reference_population=self.reference_population,rows=self._rows,pending=self._pending,last_model_version=self._last_model_version,committed_batches=self._committed_batches))
 @classmethod
 def from_state_dict(cls,state):
  s=_copy(state)
  if s['schema']!=SCHEMA:raise ValueError('Unknown detached score-window schema')
  obj=cls(s['capacity'],s['target_k'],s['reference_population']);obj._rows=[_entry(r)for r in s['rows']]
  if len(obj._rows)>obj.capacity or len({r['qid']for r in obj._rows})!=len(obj._rows):raise ValueError('Invalid saved unique-question FIFO')
  obj._last_model_version=None if s['last_model_version']is None else _integer(s['last_model_version'],'last_model_version');obj._committed_batches=_integer(s['committed_batches'],'committed_batches');obj._pending=s['pending']
  if obj._pending is not None:
   obj._pending['current_entries']=[_entry(r)for r in obj._pending['current_entries']];step=_integer(obj._pending['pre_update_model_version'],'pre_update_model_version');qids=[r['qid']for r in obj._pending['current_entries']]
   if len(set(qids))!=len(qids)or set(qids)&{r['qid']for r in obj._rows}or any(r['scored_step']!=step for r in obj._pending['current_entries']):raise ValueError('Invalid saved prepared view')
   _ages(obj._rows,step)
  elif obj._last_model_version is not None:_ages(obj._rows,obj._last_model_version)
  if obj.state_dict()!=s:raise ValueError('State restoration must preserve exact JSON metadata and FIFO order')
  return obj
