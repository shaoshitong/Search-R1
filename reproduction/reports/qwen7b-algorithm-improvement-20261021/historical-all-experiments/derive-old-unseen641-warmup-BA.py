"""One CPU-only summary of original641-Mixed NQ2048 saved EMA predictions."""
import functools,gzip,hashlib,json,math,operator,time
from pathlib import Path
REFS = [{'epoch': 1, 'weight_key': 'ema', 'predictions': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020/results/epoch-0001-predictions.jsonl.gz', 'sha256': 'aebb0a3034be84f00dd708e7a4cb13f3793164afdda3ae68efe249670ebfc9f2', 'bytes': 131795}, 'checkpoint': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/probe/epoch-0001.pt', 'sha256': 'a6a7a255588b50d65860710c513203de341a8b7f686b9ecbf18717bb6dca716b'}, 'expected_top': 169}, {'epoch': 2, 'weight_key': 'ema', 'predictions': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020/results/epoch-0002-predictions.jsonl.gz', 'sha256': '408eef7e1337991ca1feb808c85030998c46039ce496c5a2b4b319f3a8ccff61', 'bytes': 132163}, 'checkpoint': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/probe/epoch-0002.pt', 'sha256': '893e2d71e9b45abb92a9573a6091a1a0fb0a5f7ffd3e4e1201d12061317095ad'}, 'expected_top': 181}, {'epoch': 3, 'weight_key': 'ema', 'predictions': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020/results/epoch-0003-predictions.jsonl.gz', 'sha256': 'ae84774c1f033a47ac0558402b74a0a4865b2580c4938345a820b72782fb5d99', 'bytes': 132394}, 'checkpoint': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/probe/epoch-0003.pt', 'sha256': '9ef2385a29dd5b970f7ca11fe27ee2b31e2b77c23b769b43fbadaa6162c2186d'}, 'expected_top': 177}, {'epoch': 4, 'weight_key': 'ema', 'predictions': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020/results/epoch-0004-predictions.jsonl.gz', 'sha256': '467f5819147aca02b7086c4c6c345e6371caf107d61bfd35125e527cf495f7a7', 'bytes': 132583}, 'checkpoint': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/probe/epoch-0004.pt', 'sha256': 'c559ab711422019d95e805dedc375d756eff551a7a8ff9e751f40cd8af51f75f'}, 'expected_top': 178}]
base=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020')
def proof(path):
 path=Path(path);raw=path.read_bytes();return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
bank=base/'bank/complete.json';bankproof=proof(bank)
assert bankproof['sha256']=='5876bd9607b1e956d8752e13c6e54def5d77be683a940af5f03b4f0db176a64c'
manifest=json.loads(bank.read_text());groupspath=base/'bank/groups.json';groupsproof=proof(groupspath)
assert groupsproof['sha256']==manifest['groups_sha256']
groups=json.loads(groupspath.read_text())['selection_validation'];assert len(groups)==2048
ordered=hashlib.sha256(json.dumps([row['qid']for row in groups]).encode()).hexdigest()
assert ordered==manifest['ordered_qid_sha256']=='02a3dad1b18eff01915dad654934510de148fa938658b40d32920104862baf0b'
by_id={row['qid']:row for row in groups};assert len(by_id)==2048
identity=hashlib.sha256(json.dumps({q:(v['trajectory_ids'],v['labels'])for q,v in by_id.items()},sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert sum(0<sum(row['labels'])<5 for row in groups)==641
results=[]
for ref in REFS:
 assert ref['weight_key']=='ema';actual=proof(ref['predictions']['path']);assert actual==ref['predictions']
 rows=[];seen=set();tp=tn=fp=fn=0;max_error=0.
 with gzip.open(actual['path'],'rt')as stream:
  for line in stream:
   row=json.loads(line);qid=row['qid'];assert qid not in seen;seen.add(qid);truth=by_id[qid]
   assert row['trajectory_ids']==truth['trajectory_ids']and row['true_EM']==truth['labels']
   ps=row['p_j'];assert len(ps)==5 and all(math.isfinite(p)and 0<=p<=1 for p in ps)
   q=max(0.,min(1.,1-functools.reduce(operator.mul,ps,1)-functools.reduce(operator.mul,(1-p for p in ps),1)))
   max_error=max(max_error,abs(q-row['predicted_mixed']));assert max_error<1e-12
   y=0<sum(row['true_EM'])<5;pred=row['predicted_mixed']>=.5
   tp+=bool(pred and y);fn+=bool(not pred and y);fp+=bool(pred and not y);tn+=bool(not pred and not y)
   rows.append((row['predicted_mixed'],qid,y))
 assert seen==set(by_id)and tp+fn==641 and fp+tn==1407
 selected=sorted(rows,key=lambda v:(-v[0],v[1]))[:512];top=sum(v[2]for v in selected);assert top==ref['expected_top']
 results.append(dict(epoch=ref['epoch'],state='EMA0.99',checkpoint=ref['checkpoint'],predictions=actual,
  pool_questions=2048,pool_Mixed=641,threshold=.5,threshold_fitted=False,confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn),
  mixed_balanced_accuracy=.5*(tp/641+tn/1407),top_mixed=top,top_k=512,max_q_formula_error=max_error,
  exact_old_question_own_G5_EM_identity_sha256=identity))
print(json.dumps(dict(status='ORIGINAL_OLD_UNSEEN641_POOL_SAVED_EMA_Q_CPU_CONFUSION_DERIVED',time=time.time(),results=results,
 bank=bankproof,groups=groupsproof,ordered_qids_sha256=ordered,own_G5_identity_sha256=identity,
 no_new_forward=True,model_or_optimizer_loaded=False,raw_predictions_returned=False,final4096_read=False)))
