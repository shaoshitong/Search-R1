"""Pure metadata mapping for the accepted two-segment frozen-E1 cache.

No tensor/model load or output write. ``held`` is the fixed own-newseed-G5
record list. The old physical order is fit30016,held5120; added23296 follows
in a separate file, regardless of the new logical fit53312,held5120 order.
"""
import hashlib,json

def digest(v):
 return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def layout(old_cohort,expanded_fit,groups,held,*,old_fit_count=30016,holdout_count=5120,expanded_fit_count=53312):
 old_fit=list(old_cohort['fit_qids']);old_held=list(old_cohort['holdout_qids'])
 fit=[g['qid']if isinstance(g,dict)else g for g in expanded_fit]
 if len(old_fit)!=old_fit_count or len(old_held)!=holdout_count or len(fit)!=expanded_fit_count or fit[:old_fit_count]!=old_fit:
  raise ValueError('Exact old fit prefix/new fit/held budgets required')
 old_qids=old_fit+old_held;added=fit[old_fit_count:]
 if len(set(old_qids))!=len(old_qids)or len(set(fit))!=len(fit)or set(fit)&set(old_held)or set(added)&set(old_qids):
  raise ValueError('Question duplication or held/added intersection')
 by={g['qid']:g for g in groups};fixed={g['qid']:g for g in held}
 if len(by)!=len(groups)or set(by)!=set(old_qids+added)or len(fixed)!=holdout_count or len(held)!=holdout_count or set(fixed)!=set(old_held):
  raise ValueError('Exact unique accepted group/held coverage required')
 hashes=set();tids=set()
 for q in old_qids+added:
  g=by[q];ids=g['trajectory_ids'];ys=g['labels'];h=g['question_hash']
  if len(h)!=64 or h in hashes or len(ids)!=10 or len(set(ids))!=10 or tids.intersection(ids)or len(ys)!=10 or any(type(y)is not int or y not in(0,1)for y in ys):
   raise ValueError('Authentic unique question hashes/ten own IDs/strictEM required')
  hashes.add(h);tids.update(ids)
 for q in old_held:
  g,f=by[q],fixed[q]
  if f['trajectory_ids']!=g['trajectory_ids'][5:]or f['labels']!=g['labels'][5:]or f['question_hash']!=g['question_hash']or f.get('source_indices')!=[5,6,7,8,9]:
   raise ValueError('Frozen own-newseed G5 changed')
 old_ids=[tid for q in old_qids for tid in by[q]['trajectory_ids']]
 added_ids=[tid for q in added for tid in by[q]['trajectory_ids']]
 loc={q:(0,10*i)for i,q in enumerate(old_qids)}
 loc.update({q:(1,10*i)for i,q in enumerate(added)})
 width=20*3584*2
 return dict(old_qids=old_qids,added_qids=added,qid_locations=loc,
  old_trajectory_ids_sha256=digest(old_ids),new_trajectory_ids_sha256=digest(added_ids),
  physical_trajectory_ids_sha256=digest(old_ids+added_ids),
  old_trajectories=len(old_ids),new_trajectories=len(added_ids),total_trajectories=len(tids),
  old_hidden_bytes=len(old_ids)*width,new_hidden_bytes=len(added_ids)*width,
  total_hidden_bytes=len(tids)*width,batch320_hidden_bytes=320*width,
  hidden_dtype='bfloat16',question_trajectories=10,hidden_tokens=20,hidden_dimension=3584)

def locations(cache_layout,qids):
 """Return physical segment and starting-row for each requested real question."""
 if len(set(qids))!=len(qids):raise ValueError('Duplicate qids in one requested batch')
 return [cache_layout['qid_locations'][q]for q in qids]
