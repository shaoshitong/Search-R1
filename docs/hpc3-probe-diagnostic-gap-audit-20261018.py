import json,shlex,subprocess
from pathlib import Path
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT
code=r'''
import json,hashlib,time,gzip,statistics,collections,math
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3')
def rd(p):return json.loads(Path(p).read_text())
def rows(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return [json.loads(x) for x in f]
def ds(rs):return dict(collections.Counter(str(r.get('data_source')) for r in rs))
def stat(v):return dict(n=len(v),min=min(v),mean=statistics.mean(v),max=max(v)) if v else None
out=dict(epoch=time.time(),rounds={})
for n in (50,60):
 root=b/'outers'/('outer-%06d'%n)
 data=(b/'probe-refreshes'/('round-%06d'%n)/'outers'/root.name) if n==50 else root
 diag=data/'diagnostic/outers'/root.name
 report={}
 sets={}
 for split,r in [('train',data),('diagnostic',diag)]:
  qs=rd(r/'questions.json')['rows'];pf=rd(r/'prefix/complete.json');full=rd(r/('full-pool' if split=='train' else 'full')/'complete.json');scan=rd(Path(pf['scan_path'])/'scan-complete.json');index=rows(Path(pf['scan_path'])/'index.jsonl');own=rows(full['own_em']['path'])
  qmap={q['qid']:q for q in qs};byq=collections.defaultdict(list)
  for item in own:byq[item['qid']].append(item['strict_em'])
  sets[split]=set(qmap)
  perds={}
  for dataset in ds(qs):
   chosen=[key for key,q in qmap.items() if str(q.get('data_source'))==dataset];ks=[sum(byq[key]) for key in chosen]
   perds[dataset]=dict(questions=len(chosen),K_hist=dict(collections.Counter(ks)),mixed_fraction=sum(0<k<5 for k in ks)/len(ks),trajectory_em=sum(ks)/(len(ks)*5))
  report[split]=dict(actor_version=pf['actor_version'],hidden_feature_actor_version=scan['hidden_feature_actor_version'],full_actor_version=full['actor_version'],actor_model_dir=rd(r/'actor-binding.json').get('actor_model_dir'),head_max_tokens=scan['head_max_tokens'],max_policy_prefix_tokens=scan['max_policy_prefix_tokens'],questions=len(qs),data_sources=ds(qs),label_summary=perds,question_length=stat([x['initial_context']['request_length'] for x in index]),policy_length=stat([x['prefix_length'] for x in index]),index_keys=list(index[0]),feature_ranks=dict(collections.Counter(x['feature_rank'] for x in index)),sampling_flags={k:scan.get(k) for k in ['sampling','temperature','top_p','actor_hidden_native','independent_encoder_forward']})
 report['qid_overlap']=len(sets['train']&sets['diagnostic'])
 pred={}
 for split in ['train','test']:
  rs=rows(root/'probe-update'/(split+'-epoch-001-group-predictions.jsonl'));pred[split]=dict(keys=list(rs[0]))
 report['prediction_format']=pred
 # Compare recorded pre-fit score logits with actual fit-before metrics.
 score_file=root/'probe-score/candidate-scores.jsonl'
 if n==50:score_file=data/'probe-score/candidate-scores.jsonl'
 score_rows=rows(score_file)
 own=rows(data/'full-pool/own-em.jsonl');labs={r['trajectory_id']:r['strict_em'] for r in own}
 correct=0;tp=fp=tn=fn=0;gb=tb=0;means=[];perds=collections.defaultdict(list)
 source_map={r['qid']:str(r['data_source']) for r in rd(data/'questions.json')['rows']}
 for row in score_rows:
  z=row['trajectory_logits'];ps=row['trajectory_probabilities'];q=row['q'];ys=[labs[t] for t in row['trajectory_ids']];y=int(0<sum(ys)<5);positive=q>=.5
  tp+=int(positive and y);fp+=int(positive and not y);tn+=int(not positive and not y);fn+=int(not positive and y)
  gb+=-math.log(max(1e-300,q if y else 1-q));tb+=sum(max(t,0)-t*yj+math.log1p(math.exp(-abs(t))) for t,yj in zip(z,ys))/5
  means.append(q)
  perds[source_map[row['qid']]].append((q,y,int(positive==y)))
 report['pre_fit_candidate_score_recomputed']=dict(mean_q=statistics.mean(means),accuracy=(tp+tn)/len(score_rows),tp=tp,fp=fp,tn=tn,fn=fn,trajectory_bce=tb/len(score_rows),group_bce=gb/len(score_rows),total_loss=(tb+gb)/len(score_rows),dataset={k:dict(mean_q=statistics.mean(r[0] for r in rs),accuracy=statistics.mean(r[2] for r in rs),n=len(rs)) for k,rs in perds.items()})
 seen=set()
 for old in range(10,n,10):
  candidates=[b/'outers'/('outer-%06d'%old)/'questions.json',Path(str(b).replace('-r3','-r2'))/'outers'/('outer-%06d'%old)/'questions.json']
  prev=next((p for p in candidates if p.exists()),None)
  if prev:seen.update(q['qid'] for q in rd(prev)['rows'])
 if n==60:
  prev=b/'probe-refreshes/round-000050/outers/outer-000050/questions.json'
  seen.update(q['qid'] for q in rd(prev)['rows'])
 report['prior_periodic_question_overlap']=dict(prior_unique=len(seen),train_seen=len(sets['train']&seen),diagnostic_seen=len(sets['diagnostic']&seen),scope='retained_P10_full_pool_and_prefix20_refresh; excludes_initial_warmup')
 out['rounds'][str(n)]=report
proof=b/'preparation/stopped-joint-000061-v1-full-sha-cpu.json'
progress=proof.with_name(proof.name+'.progress.json');failure=proof.with_name(proof.name+'.failure.json')
out['cp61_cpu']=dict(proof_exists=proof.exists(),progress=rd(progress) if progress.exists() else None,failure=rd(failure) if failure.exists() else None)
print(json.dumps(out))
'''
addr=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip();r=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=addr,timeout=30)
if r.returncode:print(r.stdout.decode());raise RuntimeError(r.stderr.decode())
v=json.loads(r.stdout);(Path(OUTPUT)/'probe-fit50-60-diagnostic-distribution-audit.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,ensure_ascii=False))
