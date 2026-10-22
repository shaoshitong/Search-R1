"""Independent stdlib paired metric recomputation for natural/conditional G5.

Reads only the fixed training2048 and previously viewed validation2048 logits.
Never loads a model/checkpoint/tensor or untouched final-confirmation pool.
Matches qid, own five trajectory IDs and true EM before comparing any metric.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
CONDITIONAL=A/'online-mixed-layer-prefix-20261022-2gpu-v1'
NATURAL=A/'online-natural-g5-8192-20261022-v1'
ROLES=('fixed_train2048','fixed_validation2048')


def read(path):return json.loads(Path(path).read_text())
def proof(path):
 path=Path(path);raw=path.read_bytes();assert len(raw)<16*1024**2 and path.suffix!='.pt'
 return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def bce(logit,target):return max(logit,0)-logit*int(target)+math.log1p(math.exp(-abs(logit)))


def load_predictions(path):
 proof(path);rows={};tids=set()
 with gzip.open(path,'rt')as stream:
  for line in stream:
   row=json.loads(line);qid=row['qid'];own=row['trajectory_ids'];labels=row['labels']
   assert qid not in rows and len(own)==len(set(own))==len(labels)==5 and not tids.intersection(own)
   assert all(type(y)is int and y in(0,1)for y in labels)
   assert type(row['true_mixed'])is bool and row['true_mixed']==(0<sum(labels)<5)
   assert all(math.isfinite(row[name])for name in('raw_logit','EMA_logit'))
   rows[qid]=row;tids.update(own)
 assert len(rows)==2048
 return rows


def identity(rows):return {qid:(row['trajectory_ids'],row['labels'],row['true_mixed'])for qid,row in rows.items()}


def metrics(rows,label):
 values=list(rows.values());tp=tn=fp=fn=0;positive=0;loss=0.;classes={'all_wrong':[],'Mixed':[],'all_correct':[]}
 for row in values:
  y=row['true_mixed'];z=row[label+'_logit'];pred=z>=0;positive+=y;tp+=bool(pred and y);tn+=bool(not pred and not y);fp+=bool(pred and not y);fn+=bool(not pred and y);loss+=bce(z,y)
  k=sum(row['labels']);classes['all_wrong'if k==0 else'all_correct'if k==5 else'Mixed'].append((z,y,pred))
 negative=len(values)-positive;assert positive and negative
 top=sorted(values,key=lambda row:(-row[label+'_logit'],row['qid']))[:512]
 breakdown={}
 for name,own in classes.items():
  count=len(own);called=sum(pred for _,_,pred in own)
  breakdown[name]=dict(questions=count,predicted_Mixed=called,
   error_count=count-called if name=='Mixed'else called,
   error_rate=(count-called if name=='Mixed'else called)/count if count else None,
   Mixed_BCE=sum(bce(z,y)for z,y,_ in own)/count if count else None)
 return dict(questions=len(values),binary_Mixed_BCE=loss/len(values),balanced_accuracy=.5*(tp/positive+tn/negative),
  Mixed_recall=tp/positive,nonMixed_false_positive_rate=fp/negative,confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn),
  classwise=breakdown,selected512=dict(questions=512,Mixed=sum(row['true_mixed']for row in top),
   Mixed_fraction=sum(row['true_mixed']for row in top)/512,selected_qids_sha256=digest([row['qid']for row in top])))


def verify_producer(recomputed,producer):
 assert recomputed['questions']==producer['questions'] and recomputed['confusion']==producer['confusion']
 for own,name in(('binary_Mixed_BCE','binary_Mixed_BCE'),('balanced_accuracy','balanced_accuracy'),('Mixed_recall','Mixed_recall'),('nonMixed_false_positive_rate','nonMixed_false_positive_rate')):
  assert abs(recomputed[own]-producer[name])<1e-8,(own,recomputed[own],producer[name])
 assert recomputed['selected512']['Mixed']==producer['selected512']['Mixed']
 for name in('all_wrong','all_correct'):
  assert recomputed['classwise'][name]['questions']==producer[name]['questions']
  assert recomputed['classwise'][name]['predicted_Mixed']==producer[name]['predicted_Mixed']


def paired(conditional,natural,label):
 assert identity(conditional)==identity(natural),'qid/own G5/EM mismatch: not a paired comparison'
 out={}
 for name,target in(('Mixed',True),('nonMixed',False)):
  own=[q for q,row in conditional.items()if row['true_mixed']is target]
  improved=worsened=both_right=both_wrong=0
  for q in own:
   left=(conditional[q][label+'_logit']>=0)==target;right=(natural[q][label+'_logit']>=0)==target
   improved+=bool(not left and right);worsened+=bool(left and not right);both_right+=bool(left and right);both_wrong+=bool(not left and not right)
  out[name]=dict(questions=len(own),conditional_wrong_natural_right=improved,conditional_right_natural_wrong=worsened,
   both_right=both_right,both_wrong=both_wrong,recall_difference=(improved-worsened)/len(own))
 left=sorted(conditional,key=lambda q:(-conditional[q][label+'_logit'],q))[:512]
 right=sorted(natural,key=lambda q:(-natural[q][label+'_logit'],q))[:512]
 gained=set(right)-set(left);lost=set(left)-set(right)
 out['Top512']=dict(selected_overlap=len(set(left)&set(right)),newly_selected=len(gained),dropped=len(lost),
  newly_selected_Mixed=sum(natural[q]['true_mixed']for q in gained),dropped_Mixed=sum(conditional[q]['true_mixed']for q in lost),
  exact_Mixed_count_difference=sum(natural[q]['true_mixed']for q in right)-sum(conditional[q]['true_mixed']for q in left))
 return out


def compare(layers=(21,27),epochs=(0,1,2,3),require_complete=False):
 fixture=read(NATURAL/'fixture/fixture-complete.json');source=read(CONDITIONAL/'fixture-reuse.json')
 assert fixture['source_fixture']==source['fixture'] and fixture['seed']==2026102207
 assert fixture['all_layers_same_new_plans']and fixture['original_8192_qids_and_three_epoch_orders_preserved']
 results={};missing=[];refs={};identities={};initial_logits={}
 for layer in layers:
  name='layer%d-prefix20'%layer;results[name]={};refs[name]={}
  for epoch in epochs:
   files=[suite/name/('evaluation-epoch-%02d.json'%epoch)for suite in(CONDITIONAL,NATURAL)]
   needed=files+[suite/name/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role))for suite in(CONDITIONAL,NATURAL)for role in ROLES]
   absent=[str(path)for path in needed if not path.exists()]
   if absent:missing.extend(absent);continue
   evaluations=[read(path)for path in files]
   assert all(value['epoch']==epoch and value['cursor']==64*epoch and value['layer']==layer and value['prefix']==20 for value in evaluations)
   results[name][str(epoch)]={};refs[name][str(epoch)]={}
   for role in ROLES:
    predictions=[load_predictions(suite/name/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role)))for suite in(CONDITIONAL,NATURAL)]
    for rows in predictions:
     if role in identities:assert identities[role]==identity(rows),'Fixed pool changed between epochs/layers/schemes'
     else:identities[role]=identity(rows)
    metrics_by={};pairs={}
    for label in('raw','EMA'):
     own=[metrics(rows,label)for rows in predictions]
     for recomputed,value in zip(own,evaluations):verify_producer(recomputed,value['metrics'][role][label])
     metrics_by[label]=dict(conditional=own[0],natural=own[1],delta=dict(
      balanced_accuracy=own[1]['balanced_accuracy']-own[0]['balanced_accuracy'],
      Mixed_BCE=own[1]['binary_Mixed_BCE']-own[0]['binary_Mixed_BCE'],
      Mixed_recall=own[1]['Mixed_recall']-own[0]['Mixed_recall'],
      nonMixed_false_positive_rate=own[1]['nonMixed_false_positive_rate']-own[0]['nonMixed_false_positive_rate'],
      Top512_Mixed=own[1]['selected512']['Mixed']-own[0]['selected512']['Mixed'],
      Top512_Mixed_fraction=own[1]['selected512']['Mixed_fraction']-own[0]['selected512']['Mixed_fraction']))
     pairs[label]=paired(predictions[0],predictions[1],label)
     if epoch==0:initial_logits[name+'-'+role+'-'+label]=max(abs(predictions[0][q][label+'_logit']-predictions[1][q][label+'_logit'])for q in predictions[0])
    results[name][str(epoch)][role]=dict(metrics=metrics_by,paired_classification_changes=pairs)
    refs[name][str(epoch)][role]={scheme:proof(suite/name/('epoch-%02d-%s-predictions.jsonl.gz'%(epoch,role)))for scheme,suite in(('conditional',CONDITIONAL),('natural',NATURAL))}
   refs[name][str(epoch)]['producer_evaluations']=[proof(path)for path in files]
 if require_complete:assert not missing,'Wait for all requested paired epoch files; never substitute a different pool'
 if identities.keys()>={'fixed_train2048','fixed_validation2048'}:assert not set(identities['fixed_train2048'])&set(identities['fixed_validation2048'])
 result=dict(status='NATURAL_VS_CONDITIONAL_G5_MATCHED_METRICS_COMPLETE'if not missing else'PARTIAL_MATCHED_METRICS_WAITING_FOR_REQUESTED_EPOCHS',
  metrics=results,prediction_refs=refs,missing=missing,training_fixture=proof(NATURAL/'fixture/fixture-complete.json'),
  same_8192_questions_and_epoch_orders=True,same_fixed_own_G5_questions_and_EM_across_every_compared_score=True,
  same_model_and_optimizer_online_math=True,intervention='natural label-blind own5of10; genuine labels; corresponding epoch inverse-frequency weights',
  initial_raw_EMA_max_logit_difference_by_layer_pool=initial_logits,
  fixed_training_and_independent_validation_separate=True,no_online_train_accuracy_substitution=True,
  final4096_read=False,new_model_scoring=False,checkpoint_or_tensor_payload_loaded=False,threshold_tuned=False,
  stable_improvement_or_goal_completion_not_automatically_claimed=True,source=proof(__file__),time=time.time())
 out=NATURAL/'natural-vs-conditional-matched-metrics.json';temporary=out.with_suffix('.tmp');temporary.write_text(json.dumps(result,sort_keys=True)+'\n');temporary.replace(out)
 return dict(status=result['status'],report=proof(out),missing_files=len(missing),matched_epoch_pairs=sum(len(v)for v in results.values()),final4096_read=False)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--layers',type=int,nargs='+',choices=(21,27),default=[21,27]);parser.add_argument('--epochs',type=int,nargs='+',choices=(0,1,2,3),default=[0,1,2,3]);parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
 print(json.dumps(compare(tuple(args.layers),tuple(args.epochs),args.require_complete)),flush=True)


if __name__=='__main__':main()
