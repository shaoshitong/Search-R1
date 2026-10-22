import collections,hashlib,itertools,json,math,pathlib,random,time
start=time.monotonic();base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
c=json.loads((base/'joint-mixed-direct-official-169028-b128-v1/config.json').read_text());ref=c['groups'];payload=pathlib.Path(ref['path']).read_bytes()
assert len(payload)==ref['bytes'] and hashlib.sha256(payload).hexdigest()==ref['sha256'];groups=json.loads(payload)['train'];del payload
def choose(n,k):return math.factorial(n)//(math.factorial(k)*math.factorial(n-k)) if 0<=k<=n else 0
N=len(groups);eligible=[g['qid']for g in groups if 0<sum(g['labels'])<10];E=len(eligible);r=(N/2)/E;den=choose(10,5)
def domain(g):
 for key in ('dataset','domain','source','data_source'):
  if key in g:return str(g[key])
 q=str(g['qid']).lower()
 if 'hotpot'in q:return 'HotpotQA'
 if 'nq'in q:return 'NQ'
 return q.split('/')[0].split(':')[0].split('_')[0]
def hyper(K,k):return (choose(K,k)*choose(10-K,5-k)/den)if 0<=k<=K and 0<=5-k<=10-K else 0.
counts=collections.Counter(sum(g['labels'])for g in groups);bydomain={}
for g in groups:bydomain.setdefault(domain(g),collections.Counter())[sum(g['labels'])]+=1
Krows=[]
for K,n in sorted(counts.items()):
 h=[hyper(K,k)for k in range(6)];pm=1-h[0]-h[5];pp=r if 0<K<10 else 0.
 curr=[pp*h[k]/pm if 0<k<5 and pm else (1-pp)*h[k]/(1-pm) if k in(0,5)and pm<1 else 0. for k in range(6)]
 Krows.append(dict(K10=K,questions=n,natural_pMixed=pm,forced_positive_probability=pp,natural_expected_K5=[n*x for x in h],forced_expected_K5=[n*x for x in curr],natural_homogeneous=n*(1-pm),forced_homogeneous=n*(1-pp),homogeneous_exposure_ratio=(1-pp)/(1-pm)if pm<1 else None))
def aggregate(rows):
 natural=[sum(row['natural_expected_K5'][k]for row in rows)for k in range(6)];forced=[sum(row['forced_expected_K5'][k]for row in rows)for k in range(6)]
 mixed_natural=sum(natural[1:5]);mixed_forced=sum(forced[1:5]);neg_nat=N-mixed_natural;neg_forced=N-mixed_forced
 nontrivial_nat=sum(x['natural_homogeneous']for x in rows if 0<x['K10']<10);nontrivial_forced=sum(x['forced_homogeneous']for x in rows if 0<x['K10']<10)
 return dict(natural_expected_K5=natural,forced_expected_K5=forced,natural_Mixed=mixed_natural,forced_Mixed=mixed_forced,natural_Mixed_fraction=mixed_natural/N,forced_Mixed_fraction=mixed_forced/N,natural_negative_from_G10mixed=nontrivial_nat,forced_negative_from_G10mixed=nontrivial_forced,natural_fraction_negatives_from_G10mixed=nontrivial_nat/neg_nat,forced_fraction_negatives_from_G10mixed=nontrivial_forced/neg_forced)
rng=random.Random(2026102203);forced_positive=set(rng.sample(eligible,N//2));sim={'uniform':collections.Counter(),'forced':collections.Counter()};sim_domain={}
partitions={K:{role:[ix for ix in itertools.combinations(range(10),5)if(0<sum(i<K for i in ix)<5)==role]for role in(False,True)}for K in range(11)}
for g in groups:
 K=sum(g['labels']);d=domain(g);sim_domain.setdefault(d,{'uniform':collections.Counter(),'forced':collections.Counter()});labels=g['labels'];ix=rng.sample(range(10),5);k=sum(labels[i]for i in ix);sim['uniform'][k]+=1;sim_domain[d]['uniform'][k]+=1
 # Count conditional distribution by positions of the actual own labels.
 role=g['qid']in forced_positive;positive=[i for i,y in enumerate(labels)if y];negative=[i for i,y in enumerate(labels)if not y];canonical=positive+negative;ix=rng.choice(partitions[K][role]);k=sum(labels[canonical[i]]for i in ix);sim['forced'][k]+=1;sim_domain[d]['forced'][k]+=1
result=dict(status='ACTUAL169028_OWN10_CLASSCONDITIONAL_VS_LABELBLIND_G5_METADATA_ANALYSIS',metadata=ref,questions=N,G10_mixed_eligible=E,forced_positive_probability_among_G10mixed=r,negative_G10mixed_questions_per_forced_sweep=E-N//2,K10_distribution=dict(sorted(counts.items())),K10_detail=Krows,expected=aggregate(Krows),simulation_seed=2026102203,simulation_is_not_production_training_order=True,one_seed_K5_counts={p:dict(sorted(v.items()))for p,v in sim.items()},domain_K10={d:dict(sorted(v.items()))for d,v in bydomain.items()},one_seed_domain_K5={d:{p:dict(sorted(v.items()))for p,v in val.items()}for d,val in sim_domain.items()},GPU_used=False,model_features_checkpoint_payload_opened=False,seconds=time.monotonic()-start,time=time.time())
out=base/'unique-vs-labelblind-G5-real169028-20261022.json';out.write_text(json.dumps(result,sort_keys=True)+'\n');print(json.dumps(dict(result=result,receipt=dict(path=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())),sort_keys=True))
