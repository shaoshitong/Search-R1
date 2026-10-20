import pathlib,json,gzip,hashlib,math,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1');read=lambda p:json.loads(pathlib.Path(p).read_text());done=read(b/'complete.json');base=read(b/'E4-same-HF-validation/complete.json');own=done['metrics']['selection2048FreshG5HF']
def rows(ref):
 p=pathlib.Path(ref['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def chosen(v):return {r['qid'] for r in sorted(v,key=lambda x:(-x['predicted_mixed'],str(x['qid'])))[:512]}
new=rows(own['predictions']);result={'status':'ACTUAL_FINAL_DISTINCT_PROBABILITIES_AND_CLASS_ERRORS_COMPARED','same_fixed_pool_questions':2048,'new_checkpoint':done['checkpoint'],'source_E1_EMA_equals_candidate_EMA':base['checkpoints']['E1']['ema_sha256']==done['checkpoint']['accepted_hashes']['ema'],'source_E4_EMA_equals_candidate_EMA':base['checkpoints']['E4']['ema_sha256']==done['checkpoint']['accepted_hashes']['ema'],'comparisons':{},'class_probability_diagnostics':{},'actor_updates':0,'probe_updates':0,'final_pool_read':False,'observed_epoch':time.time()}
for name,cp in base['checkpoints'].items():
 old=rows(cp['predictions']);assert [(x['qid'],x['trajectory_ids'],x['strict_em']) for x in old]==[(x['qid'],x['trajectory_ids'],x['strict_em']) for x in new]
 pd=[abs(a-b) for oldrow,newrow in zip(old,new) for a,b in zip(oldrow['p_j'],newrow['p_j'])];qd=[abs(a['predicted_mixed']-z['predicted_mixed']) for a,z in zip(old,new)];changes={}
 for label,k in [('allwrong',0),('allcorrect',5),('mixed',None)]:
  parts=[(a,z) for a,z in zip(old,new) if (0<sum(a['strict_em'])<5 if k is None else sum(a['strict_em'])==k)];changes[label]={'from_nonmixed_to_mixed':sum(a['predicted_mixed']<.5<=z['predicted_mixed'] for a,z in parts),'from_mixed_to_nonmixed':sum(z['predicted_mixed']<.5<=a['predicted_mixed'] for a,z in parts)}
 result['comparisons'][name]={'p_j_mean_abs_change':sum(pd)/len(pd),'p_j_max_abs_change':max(pd),'q_mean_abs_change':sum(qd)/len(qd),'q_max_abs_change':max(qd),'threshold_05_flips':sum((a['predicted_mixed']>=.5)!=(z['predicted_mixed']>=.5) for a,z in zip(old,new)),'top512_overlap':len(chosen(old)&chosen(new)),'class_transitions':changes}
for label,k in [('allwrong',0),('allcorrect',5),('mixed',None)]:
 part=[x for x in new if (0<sum(x['strict_em'])<5 if k is None else sum(x['strict_em'])==k)];stats={'questions':len(part),'mean_q':sum(x['predicted_mixed'] for x in part)/len(part),'mean_p_j':sum(sum(x['p_j']) for x in part)/(5*len(part)),'mean_within_group_p_j_std':sum((sum((v-sum(x['p_j'])/5)**2 for v in x['p_j'])/5)**.5 for x in part)/len(part),'all_five_p_j_exactly_equal':sum(len(set(x['p_j']))==1 for x in part),'all_five_p_j_between_045_and_055':sum(all(.45<=v<=.55 for v in x['p_j']) for x in part)}
 if k is None:stats['mean_correct_minus_wrong_p_within_mixed']=sum(sum(p for p,y in zip(x['p_j'],x['strict_em']) if y)/sum(x['strict_em'])-sum(p for p,y in zip(x['p_j'],x['strict_em']) if not y)/(5-sum(x['strict_em'])) for x in part)/len(part)
 result['class_probability_diagnostics'][label]=stats
print(json.dumps(result))
