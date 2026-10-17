from pathlib import Path
import sys,json,datetime,hashlib
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
script=r'''
import pathlib,json,random,math
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-fit-20261017-0bdddc20840b-r48-q5120-g5-mixup-lr5e5');p=b/'analysis/actual-mixup-extreme-target-conflict.json';p.parent.mkdir(exist_ok=True)
if p.exists():print(p.read_text());raise SystemExit
c=json.loads((b/'preparation/launch-config.json').read_text());s=c['probe20k_study'];arm=s['arms'][0]
m=json.loads((pathlib.Path(s['dataset_base'])/'dataset-complete.json').read_text());entry=m['blocks'][0]
labels={r['trajectory_id']:r['strict_em']for r in map(json.loads,pathlib.Path(entry['own_em']).open())}
ks={}
for row in map(json.loads,(pathlib.Path(entry['scan_path'])/'index.jsonl').open()):
 o=row['candidate_ordinal'];ks[o]=ks.get(o,0)+labels[row['trajectory_id']]
assert sorted(ks)==list(range(5120))
records=[]
for epoch in range(1,6):
 plans=json.loads((b/'arms'/arm['name']/('epoch-%02d-order.json'%epoch)).read_text());pairs=extreme=interior=large=0;sumq=sumpenalty=0.
 for step,plan in enumerate(plans,1):
  offset=0
  for rank,n in enumerate(plan['rank_questions']):
   ids=plan['question_indices'][offset:offset+n];offset+=n;rng=random.Random(s['dataset_seed']+epoch*1000003+step*8+rank)
   for a,d in plan['mix_pairs'][rank]:
    lam=rng.betavariate(arm['mixup_alpha'],arm['mixup_alpha']);pairs+=1
    if {ks[ids[a]],ks[ids[d]]}=={0,5}:
     extreme+=1;interior+=int(.2<=lam<=.8);mass=lam**5+(1-lam)**5;q=1-mass;sumq+=q;large+=int(q>.5);sumpenalty+=-arm['group_weight']*math.log(mass)
 records.append(dict(epoch=epoch,actual_planned_pairs=pairs,allcorrect_allwrong_pairs=extreme,allcorrect_allwrong_fraction=extreme/pairs,lambda_between_02_08=interior,implied_q_above_05=large,mean_implied_q_if_ownEM_soft_targets_fitted=sumq/extreme,mean_weighted_group_BCE_at_ownEM_optimum=sumpenalty/extreme))
v=dict(train_questions=5120,group_weight=5,mixup_alpha=.2,actual_saved_plans_used=True,actual_own_EM_used=True,random_lambda_seed_reconstructed_from_frozen_training_code=True,extra_model_forward=False,records=records,
 explanation='For an all-correct/all-wrong pair, both true mixed labels are0, so their convex mixed target is0. OwnEM soft targets are lambda for each of5 trajectories; fitting them gives q=1-lambda^5-(1-lambda)^5>0. Thus tied group and trajectory mixup targets cannot both reach their individual optima. This is a structural conflict, not proof of its sole causal role in observed quality.')
with p.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''
p=Path('work/probe5120-mixup-target-conflict-summary.json');v=json.loads(remote(script,str(p.with_name(p.stem+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.transport.json'))));p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(dict(actual=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),records=v['records'])))
