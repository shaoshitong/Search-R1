"""Pure-Python independent ownEM/decomposition/q/statistics qualification."""
from pathlib import Path
import json,gzip,hashlib,math,subprocess,time
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-readout-question-prefix-diagnostic-v1')
def read(p):return json.loads(Path(p).read_text())
def fp(p):
 p=Path(p);v=p.read_bytes();return dict(path=str(p),bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
def check(r):
 p=Path(r['path']);v=fp(p);assert v['sha256']==r['sha256'] and ('bytes' not in r or v['bytes']==r['bytes']);return p
def mean(x):return math.fsum(x)/len(x)
def var(x):
 m=mean(x);return mean([(z-m)**2 for z in x])
def correlation(x,y):
 a=mean(x);d=mean(y);xx=[v-a for v in x];yy=[v-d for v in y];den=math.sqrt(math.fsum(v*v for v in xx)*math.fsum(v*v for v in yy))
 return math.fsum(v*w for v,w in zip(xx,yy))/den if den else None
def equal(a,c):
 assert (a is None and c is None) or (a is not None and c is not None and abs(a-c)<1e-10*(1+abs(c))),(a,c)
def sigmoid(z):return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
def q(p):return 1-math.prod(p)-math.prod(1-x for x in p)
def product(v):
 r=1.
 for x in v:r*=x
 return r
# Login Python may predate math.prod; arithmetic is explicitly independent.
def q(p):return 1-product(p)-product(1-x for x in p)
def auc(a,y):
 n=sum(y);neg=len(y)-n;ordered=sorted(zip(a,y));i=0;s=0.
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  s+=(i+1+j)/2*sum(v for _,v in ordered[i:j]);i=j
 return (s-n*(n+1)/2)/(n*neg)
def metric(a,labels,qids):
 k=[sum(y) for y in labels];y=[0<v<5 for v in k];pred=[x>=.5 for x in a];tp=sum(t and p for t,p in zip(y,pred));tn=sum(not t and not p for t,p in zip(y,pred));fn=sum(y)-tp;fp=len(y)-sum(y)-tn;top=sorted(range(len(y)),key=lambda i:(-a[i],str(qids[i])))[:512]
 return dict(BA=.5*(tp/(tp+fn)+tn/(tn+fp)),TPR=tp/(tp+fn),TNR=tn/(tn+fp),AW_FPR=mean([int(p) for p,v in zip(pred,k) if v==0]),AC_FPR=mean([int(p) for p,v in zip(pred,k) if v==5]),AUC=auc(a,y),Top512=sum(y[i] for i in top),Top512_fraction=sum(y[i] for i in top)/512)
if not (b/'complete.json').exists():print(json.dumps(dict(status='WAITING_ACTUAL_READOUT_DIAGNOSTIC_COMPLETION')));raise SystemExit(0)
done=read(b/'complete.json');c=read(check(done['config']));groups=read(check(c['groups']));validation=read(check(c['validation_complete']));bank=read(check(validation['source_groups']))['validation'];own={g['qid']:dict(zip(g['trajectory_ids'],g['labels'])) for g in bank};assert len(groups)==2048 and c['fixed_G5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
pub=read(check(c['source_publication']))
for name,r in pub['files'].items():check(dict(r,path=str(Path(pub['code_root'])/name)))
assert done['status']=='FROZEN_E4_PEER256_QUESTION_PREFIX_READOUT_DIAGNOSTIC_COMPLETE' and done['parameters_unchanged'] and done['actor_updates']==done['probe_updates']==0 and done['flatten_width']==293*3584
reports={}
for name,result in done['models'].items():
 assert result['parameters_unchanged'] and result['questions']==2048 and result['trajectories']==10240 and result['optimizer_constructed'] is False
 with gzip.open(check(result['rows']),'rt') as f:rows=[json.loads(line) for line in f]
 with gzip.open(check(c['baselines'][name]),'rt') as f:baseline={r['qid']:r for r in map(json.loads,f)}
 assert len(rows)==len(baseline)==2048;labels=[];qids=[];p_errors=[];q_errors=[];threshold_flips=0;z={k:[] for k in ('legacy','shadow_full','question','prefix')};reconstruction=[];rounding=[];pmap={k:[] for k in z};qs={k:[] for k in z};q_error=[]
 for row,g in zip(rows,groups):
  assert row['qid']==g['qid'] and row['trajectory_ids']==g['trajectory_ids'] and row['strict_em']==g['labels']==[own[g['qid']][tid] for tid in g['trajectory_ids']]
  old=baseline[g['qid']];assert old['trajectory_ids']==g['trajectory_ids'] and old['strict_em']==g['labels'];labels.append(g['labels']);qids.append(g['qid'])
  for k in z:
   values=row[k+'_logits'];assert len(values)==5 and all(math.isfinite(v) for v in values);z[k].append(values);pmap[k].append(row['legacy_p_j'] if k=='legacy' else [sigmoid(v) for v in values]);qs[k].append(row['legacy_q'] if k=='legacy' else q(pmap[k][-1]))
  assert all(0<=p<=1 and math.isfinite(p) for p in row['legacy_p_j']);assert max(abs(sigmoid(v)-p) for v,p in zip(z['legacy'][-1],row['legacy_p_j']))<2e-7
  q_error.append(abs(row['legacy_q']-q(row['legacy_p_j'])));assert q_error[-1]<1e-5
  reconstruction.extend(abs(f-a-p) for f,a,p in zip(z['shadow_full'][-1],z['question'][-1],z['prefix'][-1]));rounding.extend(abs(f-a) for f,a in zip(z['shadow_full'][-1],z['legacy'][-1]));p_errors.extend(abs(a-p) for a,p in zip(row['legacy_p_j'],old['p_j']));q_errors.append(abs(row['legacy_q']-old['predicted_mixed']));threshold_flips+=(row['legacy_q']>=.5)!=(old['predicted_mixed']>=.5)
 assert max(reconstruction)<3e-4*(1+max(abs(v) for a in z['shadow_full'] for v in a))
 mixed=[0<sum(y)<5 for y in labels];flat_y=[v for y in labels for v in y];summary=result['summary'];component={}
 for k,values in z.items():
  flat=[v for row in values for v in row];means=[mean(row) for row in values];variances=[var(row) for row in values];ps=pmap[k];gaps=[];pgaps=[];wins=0.;pairs=0;macro=[]
  for row,p,y in zip(values,ps,labels):
   if 0<sum(y)<5:
    good=[v for v,t in zip(row,y) if t];bad=[v for v,t in zip(row,y) if not t];gap=[a-v for a in good for v in bad];w=sum(x>0 for x in gap)+.5*sum(x==0 for x in gap);wins+=w;pairs+=len(gap);macro.append(w/len(gap));gaps.append(mean(good)-mean(bad));pgaps.append(mean([v for v,t in zip(p,y) if t])-mean([v for v,t in zip(p,y) if not t]))
  measured=dict(within_question_population_variance=mean(variances),between_question_mean_variance=var(means),total_population_variance=var(flat),mean_absolute_logit=mean([abs(v) for v in flat]),RMS_logit=math.sqrt(mean([v*v for v in flat])),corr_logit_trajectory_EM=correlation(flat,flat_y),corr_probability_trajectory_EM=correlation([v for p in ps for v in p],flat_y),corr_question_mean_logit_trueMixed=correlation(means,mixed),corr_within_variance_trueMixed=correlation(variances,mixed),corr_q_trueMixed=correlation(qs[k],mixed),within_correct_minus_wrong_logit=mean(gaps),within_correct_minus_wrong_probability=mean(pgaps),within_pair_micro_AUC=wins/pairs,within_question_macro_AUC=mean(macro))
  for key,value in measured.items():equal(value,summary['components'][k][key])
  equal(measured['within_question_population_variance']/(measured['within_question_population_variance']+measured['between_question_mean_variance']) if measured['total_population_variance'] else None,summary['components'][k]['within_fraction'])
  m=metric(qs[k],labels,qids)
  for key,value in m.items():equal(value,summary['scores'][k][key])
  component[k]=dict(statistics=measured,metrics=m)
 d=summary['decomposition'];equal(max(reconstruction),d['full_FP32_minus_sum_abs_max']);equal(mean(reconstruction),d['full_FP32_minus_sum_abs_mean']);equal(max(rounding),d['shadow_FP32_minus_legacy_abs_max']);equal(mean(rounding),d['shadow_FP32_minus_legacy_abs_mean']);equal(correlation(qs['legacy'],qs['question']),d['corr_legacy_q_question_only_q']);equal(correlation(qs['legacy'],qs['prefix']),d['corr_legacy_q_prefix_only_q'])
 qflat=[v for row in z['question'] for v in row];pflat=[v for row in z['prefix'] for v in row];full=[v for row in z['shadow_full'] for v in row];qm=mean(qflat);pm=mean(pflat);cov=mean([(a-qm)*(v-pm) for a,v in zip(qflat,pflat)]);qc=[v-mean(row) for row in z['question'] for v in row];pc=[v-mean(row) for row in z['prefix'] for v in row];fc=[v-mean(row) for row in z['shadow_full'] for v in row];wcov=mean([a*v for a,v in zip(qc,pc)]);qabs=mean([abs(v) for v in qflat]);pabs=mean([abs(v) for v in pflat]);vden=var(full);wden=mean([v*v for v in fc])
 for key,value in dict(mean_abs_question_share=qabs/(qabs+pabs),mean_abs_prefix_share=pabs/(qabs+pabs),question_prefix_global_covariance=cov,signed_total_variance_question_share=(var(qflat)+cov)/vden if vden else None,signed_total_variance_prefix_share=(var(pflat)+cov)/vden if vden else None,signed_within_variance_question_share=(mean([v*v for v in qc])+wcov)/wden if wden else None,signed_within_variance_prefix_share=(mean([v*v for v in pc])+wcov)/wden if wden else None).items():equal(value,d[key])
 legacy=summary['sealed_legacy_reproduction'];equal(max(p_errors),legacy['p_abs_max']);equal(mean(p_errors),legacy['p_abs_mean']);equal(max(q_errors),legacy['q_abs_max']);equal(mean(q_errors),legacy['q_abs_mean'])
 baseline_q=[baseline[g['qid']]['predicted_mixed'] for g in groups];current_top={qids[i] for i in sorted(range(2048),key=lambda i:(-qs['legacy'][i],qids[i]))[:512]};old_top={qids[i] for i in sorted(range(2048),key=lambda i:(-baseline_q[i],qids[i]))[:512]}
 reports[name]=dict(components=component,decomposition=d,legacy_reproduction=legacy,legacy_threshold_flips_vs_sealed=threshold_flips,legacy_Top512_overlap_vs_sealed=len(current_top&old_top),legacy_q_from_serialized_p_max_error=max(q_error),rows=result['rows'],ema_sha256=result['ema_sha256'],parameters_unchanged=True)
jid=(b/'job.id').read_text().strip();r=subprocess.run(['sacct','-X','-n','-P','-j',jid,'--format=JobIDRaw,State,ExitCode,Elapsed'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);assert r.returncode==0 and r.stdout.split('|')[1:3]==['COMPLETED','0:0']
out=dict(status='FROZEN_READOUT_OWN_EM_DECOMPOSITION_Q_AND_STATISTICS_INDEPENDENTLY_VERIFIED',job_id=jid,sacct=r.stdout.strip(),complete=fp(b/'complete.json'),config=done['config'],models=reports,all_2048_ownIDs_EM_match_actual_fixedG5=True,model_or_optimizer_tensors_loaded_on_login=False,optimizer_constructed=False,actor_updates=0,probe_updates=0,diagnostic_only=True,calibration_fitted=False,untouched_final_read=False,time=time.time());(b/'independent-audit.json').write_text(json.dumps(out,sort_keys=True));print(json.dumps(out))
