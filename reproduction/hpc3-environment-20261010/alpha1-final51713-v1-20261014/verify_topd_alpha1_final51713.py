"""CPU acceptance of original200 training, final51713 predictions, and owned cleanup."""
import pathlib,json,hashlib,sys,time,math,subprocess
ROOT=pathlib.Path('/data/user/jxu957/experiments/topd-20261008/runtime')
RUN=ROOT/'logs/topd-hpc3-713717/evidence';FINAL=ROOT/'logs/topd-alpha1-final51713-recovery-713717'
sys.path.insert(0,str(ROOT/'tools/topd-instruct300-source-v1'))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 if not (FINAL/'complete.json').is_file():
  print(json.dumps({'status':'FINAL51713_NOT_YET_COMPLETE','goal_complete':False,'epoch':time.time()}));return
 import pandas as pd
 from search_r1.baselines.evaluation import aggregate_answer_metrics
 from search_r1.baselines.bapo import token_f1
 from verl.utils.reward_score.qa_em import em_check
 result=json.loads((FINAL/'complete.json').read_text());assert result['status']=='FINAL51713_ALL_FOUR_GPU_WORKERS_VERIFIED' and result['rows']==51713 and result['checkpoint_step']==200
 binding=json.loads((FINAL/'binding.json').read_text());data=ROOT/'assets/common-instruct-0b002b/final.parquet'
 expected=json.loads((ROOT.parent/'outputs/training-preparation/final51713-input-fresh-audit.json').read_text())
 assert sha(data)==binding['final_parquet_sha256']==expected['final_parquet_sha256']
 frame=pd.read_parquet(data,columns=['data_source','reward_model']);assert len(frame)==51713
 rows=[]
 for receipt in result['rank_receipts']:
  p=FINAL/('rank%d-rows.json'%receipt['rank']);assert sha(p)==receipt['sha256'];part=json.loads(p.read_text());assert len(part)==receipt['rows'];rows.extend(part)
 assert [r['row_id'] for r in rows]==list(range(51713))
 correct=0
 for i,row in enumerate(rows):
  assert row['source']==frame.iloc[i]['data_source'] and row['em'] in [0.,1.] and math.isfinite(row['f1']) and 0<=row['f1']<=1
  targets=frame.iloc[i]['reward_model']['ground_truth']['target']
  if isinstance(targets,str):targets=[targets]
  predicted=row['prediction'];em=float(em_check(predicted,targets)) if predicted is not None else 0.
  f1=max((token_f1(predicted,t) for t in targets),default=0.) if predicted is not None else 0.
  assert row['em']==em and abs(row['f1']-f1)<1e-12;correct+=int(em)
 metrics=aggregate_answer_metrics(rows);assert metrics==result['metrics']
 for source,count in expected['source_counts'].items():assert metrics['val/qa_count/'+source]==count
 checkpoints=json.loads((FINAL/'all-scheduled-checkpoints-audit.json').read_text());assert checkpoints['steps']==[5,50,100,150,200]
 for report in checkpoints['reports']:
  assert report['fresh_checkpoint_read_pass'] and len(report['ranks'])==4
  for rank in report['ranks']:assert rank['state_structure_and_finiteness_pass'] and rank['optimizer_steps']==20*report['outer'] and rank['scheduler_outer']==report['outer']
 outer=json.loads((RUN/'actual-outer-updates.json').read_text());assert outer['actual_completed_outer_steps']==list(range(1,201)) and outer['export_step']==200
 rank_audits=[]
 for rank in range(4):
  p=RUN/('actor-resource-audit/actor-updates-rank%d-outer200.json'%rank);record=json.loads(p.read_text());assert record['job']=='713717' and record['rank']==rank and record['actual_scheduler_last_epoch']==200 and len(record['updates'])==200
  assert all(all(u['input_finite'].values()) and len(u['optimizer_step_records'])==20 and len(u['micro_forward_records'])==160 and all(x['finite'] for x in u['optimizer_step_records']) for u in record['updates'])
  for completed_before_outer,u in enumerate(record['updates']):
   assert u['actual_outer_update_ordinal']==completed_before_outer+1
   expected_lr=1e-6*min(completed_before_outer/57.,1.)
   for update in u['optimizer_step_records']:
    actual_lrs=update['actual_learning_rates'];assert actual_lrs and all(math.isclose(lr,expected_lr,rel_tol=1e-12,abs_tol=1e-18) for lr in actual_lrs)
  actual_lrs=record['actual_learning_rates_after_outer'];assert actual_lrs and all(math.isclose(lr,1e-6,rel_tol=1e-12,abs_tol=1e-18) for lr in actual_lrs)
  rank_audits.append({'rank':rank,'actual_optimizer_steps':4000,'microforwards':32000,'all_actual_optimizer_LRs_match_warmup57_then_constant':True,'source_sha256':sha(p)})
 math_steps=[]
 for lineage_run,low,high in [(ROOT/'logs/topd-hpc3-712204/evidence',1,5),(RUN,6,200)]:
  with (lineage_run/'topd-math-audit/driver.jsonl').open() as f:
   for line in f:
    item=json.loads(line)
    if item['kind']!='driver_outer' or not low<=item['outer_step']<=high:continue
    assert item['run_id']==('712204' if high==5 else '713717') and item['rows']==2560 and item['unique_questions']==512 and item['teacher_targets_none']
    assert all(t['finite']==t['count'] and t['count']>0 for t in item['tensors'].values())
    math_steps.append(item['outer_step'])
 assert math_steps==list(range(1,201))
 for step in range(1,201):
  teacher_run=ROOT/'logs/topd-hpc3-712204/evidence' if step<=5 else RUN
  teacher=json.loads((teacher_run/('teacher/round-%04d.json'%step)).read_text())
  assert teacher['outer_step']==step and teacher['rows']==2560 and teacher['no_gold'] and teacher['full_vocab_normalization']==152064 and teacher['generated_tokens_used_for_supervision']==0
  assert teacher['status']=='ACTUAL_TP4_KV_RELEASE_ACK_BEFORE_STUDENT_UPDATE' and len(teacher['released_workers'])==4
  assert all(w['requires_grad_parameters']==0 and w['cache_engine_absent'] and w['gpu_cache_absent'] for w in teacher['released_workers'])
 for record in outer['records']:
  assert record['metrics']['rollout/context_mismatches']==0 and record['metrics']['rollout/trajectories']==2560
 identity=json.loads((RUN/'official-student-identity.json').read_text());assert identity['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
 for name,item in identity['official_files'].items():assert sha(ROOT/'assets/qwen2.5-3b-instruct-aa8e725'/name)==item['sha256']
 c=json.loads((ROOT/'logs/topd-hpc3-713717/config.json').read_text())
 from hydra import compose,initialize_config_dir
 with initialize_config_dir(config_dir=str(ROOT/'tools/topd-instruct300-source-v1/verl/trainer/config'),version_base=None):cfg=compose(config_name='ppo_baselines',overrides=c['training_overrides'])
 actor=cfg.actor_rollout_ref.actor
 assert cfg.trainer.n_gpus_per_node==4 and cfg.data.train_batch_size==512 and cfg.actor_rollout_ref.rollout.n_agent==5
 assert actor.ppo_mini_batch_size==128 and actor.ppo_epochs==1 and actor.ppo_micro_batch_size==16
 assert cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size==cfg.actor_rollout_ref.ref.log_prob_micro_batch_size==32
 assert actor.optim.lr==1e-6 and actor.optim.lr_warmup_steps==57 and actor.entropy_coeff==0 and cfg.algorithm.kl_ctrl.kl_coef==0
 assert actor.clip_ratio==.2 and actor.grad_clip==1 and not actor.use_kl_loss
 assert cfg.baselines.topd.alpha==1 and str(cfg.baselines.distillation.teacher_path).endswith('global_step_85')
 final_job=str(result['evaluation_job']);assert result['training_job']==713717 and result['alpha']==1
 jobs=subprocess.check_output(['sacct','-j','713717,'+final_job,'-n','-P','--format=JobID,State,ExitCode'],text=True)
 assert final_job+'|COMPLETED|0:0' in jobs and '713717|FAILED|' in jobs
 cleanup=json.loads((FINAL/('owned-cleanup-'+final_job+'.json')).read_text());assert cleanup['all_spawned_children_reaped'] and cleanup['only_exact_spawned_process_handles']
 report={'status':'ALPHA1_TOPD200_AND_FINAL51713_CPU_ACCEPTANCE_PASS','epoch':time.time(),'training_job':713717,'final_evaluation_job':int(final_job),'alpha':1,'complete_outers':200,'rank_update_audits':rank_audits,'fixed_initial_revision':identity['revision'],'final_rows':51713,'correct':correct,'EM':correct/51713,'F1':metrics['val/f1/all'],'source_counts':expected['source_counts'],'independent_prediction_rescoring_pass':True,'all_scheduled_checkpoint_read_audits_pass':True,'all_200_probability_and_frozen_teacher_round_audits_pass':True,'original_termination_error_preserved':True,'original_training_gpu_resource_release_evidence':'terminal Slurm FAILED batch and step accounting, no active allocation','final_owned_cleanup_pass':True,'original_alpha01_report_unchanged':True,'user_authorized_step5_resume_lineage':True,'final_completion_receipt_sha256':sha(FINAL/'complete.json')}
 p=ROOT.parent/'outputs/training-preparation/topd-alpha1-final-acceptance.json';p.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
