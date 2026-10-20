"""Post-freeze descriptive training-label accounting, never selection/scoring."""
import hashlib,json,math,pathlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');B=A/'peer-expanded-append-only-plan-v1'
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(r):
 if fp(r['path'])!=r:raise ValueError('Input proof changed')
 return read(r['path'])
def choose(n,k):return 0 if n<k else math.factorial(n)//math.factorial(k)//math.factorial(n-k)
def stats(gs):
 hist=[0]*11;h_old=[0]*6;h_new=[0]*6;changes=0;g5counts=[0,0,0]
 for g in gs:
  ys=g['labels'];assert len(ys)==10 and all(type(y) is int and y in (0,1) for y in ys)
  k=sum(ys);hist[k]+=1;a=sum(ys[:5]);b=sum(ys[5:]);h_old[a]+=1;h_new[b]+=1;changes+=((0<a<5)!=(0<b<5))
  g5counts[0]+=choose(10-k,5);g5counts[2]+=choose(k,5);g5counts[1]+=252-choose(10-k,5)-choose(k,5)
 n=len(gs);assert sum(g5counts)==n*252
 return dict(questions=n,K10_histogram=hist,G10_Mixed=sum(hist[1:10]),G10_Mixed_fraction=sum(hist[1:10])/n,old_G5_K_histogram=h_old,new_G5_K_histogram=h_new,old_G5_Mixed=sum(h_old[1:5]),new_G5_Mixed=sum(h_new[1:5]),old_new_G5_Mixed_label_flips=changes,uniform_all252_G5_event_counts=dict(zip(('allwrong','Mixed','allcorrect'),g5counts)),uniform_expected_G5_Mixed_fraction=g5counts[1]/(n*252))
frozen=read(B/'complete.json');plan=checked(frozen['plan']);identity=checked(frozen['identity_plan'])
assert frozen['source_checkpoint_and_loss_recipe_selected'] is False and frozen['labels_or_scores_read'] is False
ids=read(A/'peer-expanded-identity-v1/complete.json');raw=checked(ids['raw_complete']);truth=checked(raw['outputs']['groups'])['train'];by={g['qid']:g for g in truth}
groups=[]
for x in identity['groups']:
 g=by[x['qid']];assert all(g[k]==x[k] for k in ('qid','question_hash','trajectory_ids'));groups.append(g)
assert len(groups)==30016 and plan['historical_updates']==256
result=dict(status='POST_IDENTITY_FREEZE_TRAINING_LABEL_DESCRIPTIVE_COUNTS_ONLY',frozen_plan=frozen['plan'],raw_complete=ids['raw_complete'],groups=raw['outputs']['groups'],old8192=stats(groups[:8192]),additional21824=stats(groups[8192:]),all30016=stats(groups),labels_used_to_select_or_reorder=False,model_scores_read=False,validation_labels_read=False,final_read=False,training_started=False,time=time.time())
p=B/'post-freeze-class-accounting.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(result,sort_keys=True)+'\n');t.replace(p);print(json.dumps(dict(receipt=fp(p),**result),sort_keys=True))
