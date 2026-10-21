"""Metadata-only transparent phase extension; original Count screen is untouched."""
import argparse,hashlib,json,time
from pathlib import Path

EPOCHS=(0,1,2,4,8,20)
MODES=('raw','EMA')
RECIPES=('Count','binary','three_event')
BASELINE_SHA='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767'
SOURCE='805123aebf8af83a7c1c44947f39c294f7aeb624'
OBJECTIVE='e155860790cde71fc2d0edd5cfdc54ce6fe882f4'
HALVES_SHA='406d5212aa4bd86dc58df270f949db3a7defb965a6ce00429f5fa64af8880a3c'

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def proof(path):
 p=Path(path);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
 if proof(ref['path'])!={k:ref[k]for k in ('path','bytes','sha256')}:raise ValueError('Exact metadata reference changed')
 return Path(ref['path'])
def immutable(path,value):
 p=Path(path);raw=canonical(value)+b'\n';p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=raw:raise ValueError('Frozen family protocol differs')
 else:
  tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(raw);tmp.replace(p)

def eligible(candidate,baseline):
 c,b=candidate['NQ'],baseline['NQ']
 passed=c['Mixed_BA']is not None and b['Mixed_BA']is not None and c['Mixed_BA']>b['Mixed_BA']and c['Top25percent']['Mixed_count']>b['Top25percent']['Mixed_count']
 for key in ('half0','half1'):
  c,b=candidate[key],baseline[key]
  passed=passed and c['Mixed_BA']is not None and b['Mixed_BA']is not None and c['Mixed_BA']>=b['Mixed_BA']and c['Top25percent']['Mixed_count']>=b['Top25percent']['Mixed_count']
 return bool(passed)

def select(candidates,baseline):
 expected={(r,e,m)for r in RECIPES for e in EPOCHS for m in MODES}
 if len(candidates)!=36 or {(c['recipe'],c['epoch'],c['model'])for c in candidates}!=expected:raise ValueError('All36 predeclared family nodes must complete before selection')
 rows=[c for c in candidates if eligible(c,baseline)]
 rows.sort(key=lambda c:(-c['NQ']['Top25percent']['Mixed_count'],-c['NQ']['Mixed_BA'],c['epoch'],0 if c['model']=='EMA'else 1,RECIPES.index(c['recipe'])))
 return rows[0]if rows else None

def freeze(base,baseline,code_commit):
 base,baseline=Path(base),Path(baseline);target=base/'family-protocol.json'
 if target.exists():return dict(status='EXISTING_FROZEN_FAMILY_PRESERVED',protocol=proof(target))
 config_ref=proof(baseline/'config.json');c=read(config_ref['path']);pub_ref=proof(baseline/'source-publication.json');pub=read(pub_ref['path'])
 if config_ref['sha256']!=BASELINE_SHA or pub['commit']!=SOURCE or c['legacy_frozen']is not True or c['legacy_updates']or c['actor_updates']:raise ValueError('Only actual frozen805 Count baseline accepted')
 observed={str(e):(baseline/'evaluations'/('epoch-%02d'%e)/'complete.json').exists()for e in EPOCHS}
 if any(observed.values())or any((baseline/'evaluations').glob('epoch-*/*.jsonl.gz')):raise ValueError('Cannot claim before-score family registration after Count prediction outputs exist')
 halves_ref=proof(baseline/'screen-protocol/frozen-NQ-halves.json');half=read(halves_ref['path'])
 if halves_ref['sha256']!=HALVES_SHA or half['config']!=config_ref or half['labels_read']or half['scores_read']or len(half['halves'])!=2 or any(len(h)!=1280 for h in half['halves'])or half['half_SHA256']!=[digest(h)for h in half['halves']]:raise ValueError('Original pre-score NQ halves differ')
 for key in ('cohort','orders','fit_view_epochs'):checked(c[key])
 for key in ('internal_holdout_selection','internal_holdout_groups','identity_exclusion_acceptance'):checked(c['inputs'][key])
 value=dict(status='FROZEN_THREE_RECIPE36_NODE_HEAD_OBJECTIVE_FAMILY_BEFORE_COUNT_EPOCH0',phase='Transparent new family extension; original Count12-node screen/history unchanged',
  frozen_epoch=time.time(),publication_commit=code_commit,implementation=proof(__file__),baseline_config=config_ref,baseline_publication=pub_ref,
  recipes=list(RECIPES),epochs=list(EPOCHS),models=list(MODES),candidate_nodes=36,
  nodes=[dict(recipe=r,epoch=e,model=m)for r in RECIPES for e in EPOCHS for m in MODES],
  objective_source_commit=OBJECTIVE,Count_source_commit=SOURCE,original_Count_protocol_unchanged=True,
  frozen_NQ_halves=halves_ref,half_ID_SHA256=half['half_SHA256'],identity_selection=c['inputs']['internal_holdout_selection'],own_new5_groups=c['inputs']['internal_holdout_groups'],identity_exclusions=c['inputs']['identity_exclusion_acceptance'],
  cohort=c['cohort'],original938_orders=c['orders'],fit_view_epochs=c['fit_view_epochs'],grouping=c['grouping'],
  primary_holdout_source_indices=[5,6,7,8,9],holdout_questions=5120,NQ_questions=2560,HotpotQA_questions=2560,
  fit_questions=30016,updates_per_epoch=938,epochs_total=20,update_budget=18760,head_LR=1e-4,EMA_decay=.99,questions_per_update=32,views_per_question=4,view_weight=.25,
  same_actual_cache_and_full_CP0_required=True,same_initial_head_raw_EMA_Adam0_RNG_required=True,
  same_cache_baseline='Frozen raw E1 float32 Z; float64 stable sigmoid/product on exact own-new5; no extra forward',
  score=dict(Count='sum unadjusted softmax6[K1..4]',binary='unadjusted event-softmax Mixed; tau-reference balanced decision score, not calibrated original-distribution probability',three_event='unadjusted event-softmax Mixed; tau-reference balanced decision score, not calibrated original-distribution probability',threshold=.5,inference_train_prior_offset_applied=False),
  gate='NQ BA and Top640 strictly exceed same-cache frozen E1 product-q; each fixed1280 NQ half BA and Top320 noninferior',
  ranking_coverage=.25,tie_order=['NQ Top640 descending','NQ BA descending','earlier epoch','EMA before raw','recipe Count then binary then three_event'],
  all36_required_before_selection=True,external_candidate_budget_across_entire_family=1,historical_Count_screen_does_not_bypass_new_family_budget=True,external_scoring_permitted=False,
  external_requires_separate_loaded_candidate_online_micro8_qualification_and_root_GO=True,Hotpot_and_pooled_reported_but_not_gate=True,
  final_payload_read=False,external_payload_opened=False,Count_completed_nodes_at_freeze=observed,Count_cache_complete_at_freeze=(baseline/'cache/complete.json').exists(),Count_CP0_at_freeze=(baseline/'checkpoints/cursor-000000.pt').exists(),
  GPU_submitted=False,model_or_checkpoint_tensor_loaded=False)
 immutable(target,value)
 return dict(status=value['status'],protocol=proof(target),candidate_nodes=36,completed_Count_nodes_at_freeze=0,half_ID_SHA256=half['half_SHA256'],external_candidate_budget=1,GPU_submitted=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--baseline',required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
 print(json.dumps(freeze(a.base,a.baseline,a.code_commit),sort_keys=True),flush=True)
if __name__=='__main__':main()
