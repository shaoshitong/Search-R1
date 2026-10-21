"""Small coarse-output schema adapter; original Count auditor is unchanged.

Use only SHA-verified internal5120 rows/own-new5 refs. Learned six latent slots
are not asserted to identify six true count probabilities. Metrics reuse the
same independent stdlib Count implementation; no model or fitting occurs.
"""
import gzip,json,math
from . import direct_count_large_independent_audit as common

def probabilities(values,n):
 if len(values)!=n or any(not isinstance(x,(float,int))or isinstance(x,bool)or not math.isfinite(x)or not 0<=x<=1 for x in values)or abs(sum(values)-1)>1e-6:raise ValueError('Finite normalized probability vector required')

def validate_row(row,own,recipe,semantics,frozen_p=None):
 if recipe not in ('binary','three_event')or not isinstance(semantics,str)or 'tau-reference balanced decision score'not in semantics or 'not calibrated original-distribution Mixed probability'not in semantics:raise ValueError('Explicit coarse reference-score semantics required')
 if row['trajectory_ids']!=own['trajectory_ids']or row['strict_em']!=own['labels']or type(row['K'])is not int or row['K']!=sum(own['labels'])or common.normalized_domain(row['domain'])!=common.normalized_domain(own['domain']):raise ValueError('Real own-new5 IDs/EM/domain mismatch')
 six,events=row['unadjusted_six_count_probabilities'],row['unadjusted_event_probabilities'];probabilities(six,6);probabilities(events,2 if recipe=='binary'else 3)
 expected=[six[0]+six[5],sum(six[1:5])]if recipe=='binary'else[six[0],sum(six[1:5]),six[5]]
 if any(abs(a-b)>1e-6 for a,b in zip(events,expected))or not math.isfinite(row['q'])or abs(row['q']-events[1])>1e-6:raise ValueError('Unadjusted event/six-latent Mixed score differs')
 p=row['p_j']
 if len(p)!=5 or any(not math.isfinite(x)or not 0<=x<=1 for x in p):raise ValueError('Five finite frozen legacy trajectory probabilities required')
 if frozen_p is not None and (len(frozen_p)!=5 or any(abs(a-b)>1e-6 for a,b in zip(p,frozen_p))):raise ValueError('Frozen native-logit trajectory probabilities differ')
 if not math.isfinite(row['coarse_event_NLL'])or row['coarse_event_NLL']<0:raise ValueError('Finite optimized coarse-event NLL required')
 common.metrics([row])
 return row

def audit_stream(path,qids,fixed,recipe,semantics,halves,frozen_probabilities=None):
 rows=[];seen=set()
 with gzip.open(path,'rt')as f:
  for line in f:
   row=json.loads(line);qid=row['qid']
   if qid in seen or qid not in fixed:raise ValueError('Duplicate/foreign internal qid')
   seen.add(qid);validate_row(row,fixed[qid],recipe,semantics,None if frozen_probabilities is None else frozen_probabilities[qid]);rows.append(row)
 if [r['qid']for r in rows]!=qids or seen!=set(qids):raise ValueError('Exact frozen own-new5 order/coverage required')
 return dict(status='COARSE_INTERNAL_OWN_NEW5_AND_SCORE_SCHEMA_VERIFIED',recipe=recipe,score_semantics=semantics,
  metrics=common.partitions(rows,halves),own_IDs_and_strict_EM_joined=True,six_latent_internal_count_split_claim=False,
  count_NLL_or_six_way_accuracy_used_as_correctness_evidence=False,parameters_or_thresholds_fitted=False,
  optimized_coarse_event_NLL_mean=sum(r['coarse_event_NLL']for r in rows)/len(rows),
  predictions=common.proof(path),rows_returned=False,external_or_final_payload_opened=False)
