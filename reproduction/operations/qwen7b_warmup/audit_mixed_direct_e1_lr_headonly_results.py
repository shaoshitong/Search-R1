"""Small existing-output Mixed metrics and training curve audit; no new scoring."""
import hashlib,json,math,os,subprocess,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
BASE=A/'mixed-direct-e1-lr-headonly-20261022-v1';OUT=A/'mixed-direct-e1-lr-headonly-results-v1'
CONFIG_SHA='662f0931dabdcaace9d645d7c0fded786a9fe34cbe1421fdfd557b450803caf7'
PRIOR=dict(path=str(A/'mixed-direct-b128-four-epoch-metrics-v1/independent-metrics.json'),bytes=16601,sha256='a5bea14d74ebe851c8fbc99da20cbced8c8ea9bcbf7549e39bb03ff9ed0873a5')
PREDICTION_SHA={'2':{'raw':'193166e6e5b7a929ac74b928b89ec45098c718662984234ec557d65c703e7e3b','EMA':'ea3792905711438f4dae0d1cf882bce0915506e40e1ce09ca135fc3efc0bbc86'},'3':{'raw':'42d2eb6c1fdc7c1f04837204bbe5d75acec25ea6d353f63bc646fd2ffb130b22','EMA':'822fde013924812433f9d2d75cca85a697f808fe46cea6008b53fc2aee150f58'}}

def curve_summary(h):
 p=BASE/'curve.jsonl';assert p.stat().st_size<32*1024**2;raw=p.read_bytes();assert raw.endswith(b'\n');rows=[json.loads(x)for x in raw.splitlines()if x.strip()];assert len(rows)==2642
 previous=2577
 for r in rows:
  cursor=r['cursor'];assert cursor==previous+1 and r['source_cursor']==2577 and r['head_steps']==cursor and r['native_steps']==min(cursor,3898);previous=cursor
  assert r['questions']==r['effective_batch_questions']==128 and r['views']==512 and r['mixed_views']==r['nonmixed_views']==256 and r['micro_questions']==64
  c=r['confusion'];assert set(c)=={'tp','tn','fp','fn'}and all(type(v)is int and v>=0 for v in c.values())and c['tp']+c['fn']==c['tn']+c['fp']==256
  assert math.isfinite(r['binary_Mixed_BCE'])and abs(r['loss']-r['binary_Mixed_BCE'])<1e-12
  phase='phase2_joint_low_lr'if cursor<=3898 else'phase3_head_only';assert r['phase']==r['training_phase']==phase and r['native_frozen']==(cursor>3898)and r['native_lr']==5e-7 and r['head_lr']==(5e-7 if cursor<=3898 else 5e-5)
 def summary(items):
  c={k:sum(r['confusion'][k]for r in items)for k in('tp','tn','fp','fn')};pos=c['tp']+c['fn'];neg=c['tn']+c['fp'];tpr=c['tp']/pos;tnr=c['tn']/neg
  return dict(updates=len(items),first_cursor=items[0]['cursor'],last_cursor=items[-1]['cursor'],question_visits=len(items)*128,views=pos+neg,binaryMixedBCE=sum(r['binary_Mixed_BCE']*r['views']for r in items)/(pos+neg),confusion=c,Mixed_BA=(tpr+tnr)/2,Mixed_recall=tpr,nonMixed_recall=tnr,nonMixed_FPR=1-tnr,Mixed_FNR=1-tpr)
 result={}
 for epoch,lo,hi in((2,2578,3898),(3,3899,5219)):
  part=[r for r in rows if lo<=r['cursor']<=hi];assert len(part)==1321;result[str(epoch)]=dict(overall=summary(part),last100=summary(part[-100:]))
 return dict(curve=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),checks=dict(rows=len(rows),continuous_cursors=True,batch128_micro64_and_256_256_views=True,phase_native_and_head_clocks_and_LRs=True),phases=result,scope='training_sampler_windows; repeated visits; not independent validation or unique question accuracy')

def main(h):
 terminal=subprocess.check_output(['sacct','-X','-n','-P','-j','747723','--format=JobIDRaw,State,ExitCode,Elapsed']).decode().strip().splitlines();terminal=[r for r in terminal if r.split('|')[0]=='747723'];assert len(terminal)==1 and terminal[0].split('|')[:3]==['747723','COMPLETED','0:0']
 config=h.proof(BASE/'config.json');assert config['sha256']==CONFIG_SHA;finalref=h.proof(BASE/'complete.json');final=h.read(finalref['path']);assert final['status']=='E1_CONTINUATION_LOW_LR_E2_HEADONLY_E3_COMPLETE'and final['cursor']==final['head_steps']==5219 and final['native_steps']==3898 and final['final4096_scored']is False
 prior=h.read(h.checked(PRIOR));reference=h.rows_for(prior['prediction_refs']['1']['raw'],2048);identity={r['qid']:(r['trajectory_ids'],r['strict_em'])for r in reference};pool_k=h.summarize(reference)['k_histogram'];metrics={};refs={};predictions={};bce_errors=[]
 for epoch,cursor,native in((2,3898,3898),(3,5219,3898)):
  ref=h.proof(BASE/('evaluations/epoch-%02d/complete.json'%epoch));complete=h.read(ref['path']);cp=complete['checkpoint'];assert complete['status']=='DIRECT_MIXED_B128_FIXED512_2048_EVALUATION_COMPLETE'and complete['epoch']==epoch and complete['cursor']==cursor and complete['config_sha256']==CONFIG_SHA and complete['checkpoint_saved_before_evaluation']and complete['models_two_Adam_RNG_native_buffers_unchanged']and complete['final4096_scored']is False and complete['threshold_fitted']is False
  assert cp['cursor']==cp['head_steps']==cursor and cp['native_steps']==native and cp['allocation_job_id']=='747723'and cp['full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed']
  assert complete['validation_proof']==h.read(h.checked(prior['epoch_complete_refs']['1']))['validation_proof'];metrics[str(epoch)]={};predictions[str(epoch)]={};refs[str(epoch)]=ref
  for kind in('raw','EMA'):
   item=complete['metrics'][kind]['fixed_selection2048'];assert item['predictions']['sha256']==PREDICTION_SHA[str(epoch)][kind];rows=h.rows_for(item['predictions'],2048);assert{r['qid']:(r['trajectory_ids'],r['strict_em'])for r in rows}==identity
   summary=h.summarize(rows);chosen=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512];top=h.summarize(chosen);assert summary['k_histogram']==pool_k and summary['confusion']==item['pool']['confusion']and top['mixed_questions']==item['selected512']['mixed_questions']and top['k_histogram']==item['selected512']['k_histogram']
   assert item['ordered_qids_sha256']==h.digest([r['qid']for r in rows])and item['ordered_own_trajectory_ids_sha256']==h.digest([r['trajectory_ids']for r in rows])and item['selected_qids_sha256']==h.digest([r['qid']for r in chosen])
   for key in('mixed_balanced_accuracy','mixed_recall','nonmixed_recall','mixed_roc_auc'):assert summary[key]==item['pool'][key]
   error=abs(summary['binaryMixedBCE_from_probability']-item['binaryMixedBCE']);assert summary['saturated_probability_count']==0 and error<1e-5;bce_errors.append(error)
   for key,value in summary['classwise'].items():assert abs(value['binaryMixedBCE_from_probability']-item['classwise_binaryMixedBCE']['Mixed'if key=='mixed'else key]['binaryMixedBCE'])<1e-5
   summary.update(producer_stable_binaryMixedBCE=item['binaryMixedBCE'],selected512=dict(mixed=top['mixed_questions'],fraction=top['mixed_questions']/512,k_histogram=top['k_histogram']),cursor=cursor);metrics[str(epoch)][kind]=summary;predictions[str(epoch)][kind]=item['predictions']
 curve=curve_summary(h);out=dict(status='E1_LOW_LR_E2_HEADONLY_E3_FIXED2048_MIXED_METRICS_INDEPENDENTLY_RECOMPUTED',source=h.proof(__file__),metric_helpers=h.proof(h.__file__),metric_helper_origin='d47341e075091db5231307638a7f00bfe3908a12',job_id='747723',terminal=terminal[0],config=config,complete=finalref,epoch_complete_refs=refs,prediction_refs=predictions,reference_E1_independent_audit=PRIOR,reference_E1_metrics=prior['metrics']['1'],metrics=metrics,pool_K0_to_K5=pool_k,training_curve=curve,all_four_predictions_same_own_G5_IDs_and_EM=True,BCE_probability_vs_stable_max_error=max(bce_errors),threshold=.5,new_scoring=False,final4096_read=False,model_checkpoint_or_feature_payload_read=False,EM_provenance_inherited_not_recomputed_from_raw=True,producer_checkpoint_tensor_reload_inherited_not_repeated=True,epoch_time=time.time())
 OUT.mkdir(exist_ok=True);p=OUT/'independent-metrics.json'
 if p.exists():
  old=h.read(p);assert old['metrics']==out['metrics']and old['training_curve']==out['training_curve'];out=old
 else:
  with p.open('xb')as stream:stream.write((json.dumps(out,sort_keys=True,separators=(',',':'))+'\n').encode())
 print(json.dumps(dict(out,independent_audit_ref=h.proof(p)),sort_keys=True))
if __name__=='__main__':
 from reproduction.operations.qwen7b_warmup import audit_mixed_direct_b128_epochs as helpers
 main(helpers)
