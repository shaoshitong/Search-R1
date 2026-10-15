"""Synthetic scalar/held-receipt protocol tests, never remote raw acceptance."""
from contextlib import ExitStack
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import summarize_sigma3_progress as metrics
from reproduction.hpc3_qprobe import raw_audit_followthrough_sigma3 as raw


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);data=json.dumps(value).encode() if not isinstance(value,bytes) else value
    path.write_bytes(data);return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def canonical_fixture(invert=False,undefined=False):
    random=[0]*48+[1]*16;top=[0]*224+[2]*224
    if invert:random=[0]*32+[1]*32;top=[0]*336+[2]*112
    if undefined:random=[0]*64;top=[0]*448
    groups={'random_exploration':random,'predicted_diversity':top,'all':random+top}
    values={};p=[.2,.1,.1,.1,.1,.4];pred_std=.2*math.sqrt(.2)+.2*math.sqrt(.3)
    for name,ks in groups.items():
        n=len(ks);mixed=[int(0<k<5) for k in ks];stds=[metrics.STD[k] for k in ks]
        row=dict(questions=n,count_nll=sum(-math.log(p[k]) for k in ks)/n,
            mean_mae=sum(abs(.6-k/5) for k in ks)/n,std_mae=sum(abs(pred_std-s) for s in stds)/n,
            std_mse=sum((pred_std-s)**2 for s in stds)/n,mixed_brier=sum((.4-y)**2 for y in mixed)/n,
            actual_mixed_fraction=sum(mixed)/n,actual_sample_std_mean=math.fsum(stds)/n,
            actual_em_mean=sum(ks)/(5*n),all_wrong_fraction=ks.count(0)/n,all_correct_fraction=ks.count(5)/n)
        if len(set(stds))>1:row['std_pairwise_concordance']=.5
        if 0<sum(mixed)<n:row['std_mixed_auc']=.5
        values[name]=row
    ks=groups['all']
    return dict(prequential=values,K_distribution={str(k):ks.count(k) for k in range(6)},strict_correct=sum(ks))


class PrefixFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.top=Path(self.temp.name).resolve();self.base=self.top/'sigma3';self.directory=self.top/'raw'
        self.commit='a'*40;self.repository=self.top/'source'/self.commit
        self.source=dict(commit=self.commit);self.raw_source=dict(commit=metrics.RAW)
        self.stack=ExitStack()
        for target,key,value in ((metrics,'BASE',self.base),(raw,'BASE',self.base),(raw,'DIRECTORY',self.directory),
            (metrics,'__file__',str(self.repository/'reproduction/hpc3_qprobe/summarize_sigma3_progress.py')),
            (metrics.ctl,'OWNER',os.getuid()),(metrics.reader.shared,'OWNER',os.getuid())):
            self.stack.enter_context(patch.object(target,key,value))
        self.stack.enter_context(patch.object(metrics.ctl,'owner'))
        self.stack.enter_context(patch.object(metrics.ctl,'caps'))
        self.stack.enter_context(patch.object(metrics.ctl,'source',return_value=(self.repository,self.source)))
        self.stack.enter_context(patch.object(metrics.reader,'load_fixed',return_value=(raw,self.raw_source)))
        self.stack.enter_context(patch.object(raw,'validate',return_value=(self.top/'source'/metrics.RAW,self.raw_source)))
        self.stack.enter_context(patch.object(metrics.ctl,'live',return_value=False))
        self.stack.enter_context(patch.dict(os.environ,QPROBE_SOURCE=str(self.repository),PYTHONDONTWRITEBYTECODE='1'))
        self.stack.enter_context(patch.object(sys,'dont_write_bytecode',True))
        self.manifest=write(self.base/'logs/run-manifest.json',dict(run_id=metrics.RUN,source_commit=metrics.PRODUCER,
            training_world_size=4,question_probe=metrics.reader.LOSS))
        self.allocation=write(self.base/'logs/native-allocation-admission.json',dict(actual='external native binding'))
        self.producer=dict(base=str(self.base),run_id=metrics.RUN,producer_job=metrics.JOB,producer_commit=metrics.PRODUCER,
            source_manifest_sha256=self.manifest['sha256'],native_allocation_sha256=self.allocation['sha256'])
        self.parts=[dict(first=1,last=200,producer=self.producer)]
        self.supervisor=dict(pid=100,start_ticks=1000,sid=100,uid=os.getuid())
        write(self.directory/'claim.json',dict(source=self.raw_source,parts=self.parts,training_control_permitted=False,goal_complete=False))
        write(self.directory/'process.json',self.supervisor)
        self.ledger=[];completed=0
        for index,complete in enumerate((False,True,True,False),1):
            aid='%032x'%index;outer=completed+1;completed+=int(complete)
            self.ledger.append(dict(logical_update=outer,attempt_id=aid,attempt_index=index,run_id=metrics.RUN,slurm_job_id=metrics.JOB,
                optimizer_writes_this_attempt=20 if complete else 0,complete_outer=complete,opportunities_processed=20,
                scheduler_advances=int(complete),completed_outer=completed,status='complete' if complete else 'recoverable_guard'))
        self.ledger_path=self.base/'logs/actual-attempts.jsonl';self.ledger_path.write_text(''.join(json.dumps(r)+'\n' for r in self.ledger))
        self.tasks=raw.actual_tasks(self.parts)
        for task in self.tasks:self.publish_attempt(task)
    def tearDown(self):self.stack.close();self.temp.cleanup()
    def publish_attempt(self,task,*,bad_loss=False,bad_cohort=False):
        i=task['attempt_index'];aid=task['attempt_id'];step=task['step'];folder=self.directory/'attempts'/aid
        canonical=canonical_fixture(invert=step==2)
        if bad_cohort:canonical['prequential']['all']['std_mse']+=.01
        canonical.update(actual_head_committed=task['complete_outer'],actual_raw_decode_strict_terminal_em_equal=True,
            actual_512_qid_text_gold_equal=True,actual_selection_G5_E64_top448_prequential_head_equal=True,questions=512,trajectories=2560)
        event=dict(outer=step,attempt_id=aid)
        if task['complete_outer']:
            event.update(stage='question_probe_outer_committed',successful_actor_snapshot_id=metrics.RUN+':'+str(step),
                label_policy_snapshot_id='policy-'+aid,prequential=canonical['prequential'],head_update=dict(completed_outer=step,
                update_count=step,label_policy_snapshot_id='policy-'+aid,head_optimizer_updates=40+4*step,replay_groups=5120+512*step,
                sigma_ce=.8,std_mse=.2,count_ce=1.3,mean_mse=.1,replay_weighted_loss=1.13+(0.01 if bad_loss else 0)))
        else:event.update(stage='question_probe_attempt_aborted',head_updated=False,labels_observed=True)
        head=self.base/'artifacts/question-probe'/('completed' if task['complete_outer'] else 'aborted')/(aid+'.json')
        canonical['head_file']=write(head,event)
        output=dict(status='ACTUAL_RAW_CANONICAL_TRACE_ALLTOKENS_AND_ACTOR_ROWS_EQUAL',step=step,job=metrics.JOB,
            source_commit=metrics.PRODUCER,source_manifest=self.manifest,allocation_file=self.allocation,
            actual_attempt_transaction=dict(attempt_id=aid,attempt_index=i,complete_outer=task['complete_outer'],
                actual_optimizer_writes=task['actual_optimizer_writes'],attempt_record_sha256=task['record_sha256']),canonical_proof=canonical,
            actual_private_trace_byte_equal=True,actual_all2560_ordered_ids_equal=True,actual_512_parent_digests_equal=True,
            actual_2560_actor_tensor_rows_equal=True,actual_all_retained_token_contexts_equal=True,actual_selected_logprob_projection_equal=True,
            hf_forward_performed=False,model_weights_loaded=False,generated_trajectories=0,actor_or_head_updates_performed=0,evidence_files_written=0,
            raw_file=dict(sha256='b'*64),private_trace_file=dict(sha256='c'*64),included_actor_token_count=100)
        claim=write(folder/'claim.json',dict(task=task,source=self.raw_source,created_unix=1))
        child=dict(pid=200+i,start_ticks=2000+i,sid=100,uid=os.getuid());write(folder/'child.json',child)
        out=write(folder/'audit.stdout',output);err=write(folder/'audit.stderr',b'')
        result=dict(task=task,source=self.raw_source,claim_sha256=claim['sha256'],process=self.supervisor,child=child,
            actual_child_exit_code=0,started_unix=2,completed_unix=3,stdout=out,stderr=err)
        write(folder/'result.json',result)
        consumed=raw.consume_task(task,folder,self.supervisor)
        write(self.directory/'accepted'/('%06d.json'%i),consumed)
    def test_guarded_exact_prefix_real_RAW_consumer_and_weighted_aggregates(self):
        answer=metrics.summarize(self.commit)
        self.assertEqual(answer['successful_outer_prefix'],2);self.assertEqual(answer['normal_return_attempts_consumed'],4)
        self.assertEqual(answer['fully_skipped_guard_attempts'],2);self.assertFalse(answer['goal_complete'])
        groups=answer['cumulative']['cohorts'];self.assertEqual(groups['random_exploration']['mixed_questions'],16+32)
        self.assertEqual(groups['predicted_diversity']['mixed_questions'],224+112)
        self.assertEqual(groups['all']['questions'],1024);self.assertEqual(groups['all']['mixed_questions'],384)
        self.assertAlmostEqual(answer['cumulative']['top_minus_random_mixed_pp'],0.)
        self.assertAlmostEqual(answer['cumulative']['top_over_random_mixed_rate_ratio'],1.)
        self.assertNotIn('std_mixed_auc',groups['all'])
        self.assertEqual(groups['all']['std_mixed_auc_within_policy_outer_mean'],.5)
        self.assertIn('never cross-policy pooled',answer['cumulative']['ranking_scope'])
        loss=answer['per_outer'][0]['postfit_loss'];self.assertEqual(loss['normalized_sigma_mse'],.2)
        self.assertEqual(loss['head_optimizer_updates_this_fit'],44)
        self.assertNotEqual(groups['all']['prequential_sigma_mse_raw_scale'],loss['normalized_sigma_mse'])
    def test_max_outer_keeps_leading_guard_and_refuses_beyond_prefix(self):
        answer=metrics.summarize(self.commit,max_outer=1)
        self.assertEqual(answer['successful_outer_prefix'],1);self.assertEqual(answer['fully_skipped_guard_attempts'],1)
        self.assertEqual(answer['normal_return_attempts_consumed'],2)
        with self.assertRaisesRegex(ValueError,'max_outer exceeds'):metrics.summarize(self.commit,max_outer=3)
        path=self.directory/'accepted/000001.json';path.rename(path.with_name('000005.json'))
        with self.assertRaisesRegex(ValueError,'exact contiguous attempt prefix'):metrics.summarize(self.commit)
    def test_head_same_length_tamper_fresh_fullsha_rejected(self):
        task=self.tasks[1];path=self.base/'artifacts/question-probe/completed'/(task['attempt_id']+'.json')
        before=path.read_bytes();after=before.replace(b'1.13',b'1.14');self.assertEqual(len(before),len(after));self.assertNotEqual(before,after)
        path.write_bytes(after)
        with self.assertRaisesRegex(ValueError,'fresh original full SHA differs'):metrics.summarize(self.commit)
    def test_nonzero_or_live_actual_child_rejected(self):
        task=self.tasks[0];path=self.directory/'attempts'/task['attempt_id']/'result.json';original=json.loads(path.read_text())
        write(path,dict(original,actual_child_exit_code=7))
        with self.assertRaisesRegex(ValueError,'actual audit child wait/birth/task'):metrics.summarize(self.commit)
        write(path,original)
        with patch.object(metrics.ctl,'live',return_value=True),self.assertRaisesRegex(ValueError,'actual audit child wait/birth/task'):
            metrics.summarize(self.commit)
    def test_authentic_but_wrong_loss_formula_or_cohort_weighting_refuses(self):
        self.publish_attempt(self.tasks[1],bad_loss=True)
        with self.assertRaisesRegex(ValueError,'normalized sigma MSE.*formula differs'):metrics.summarize(self.commit)
        self.publish_attempt(self.tasks[1],bad_cohort=True)
        with self.assertRaisesRegex(ValueError,'cohort weighted mean differs: std_mse'):metrics.summarize(self.commit)

    def test_summary_imports_no_torch_numpy_and_writes_no_file(self):
        import builtins
        real_import=builtins.__import__
        def pure_import(name,*args,**kwargs):
            self.assertNotIn(name.split('.')[0],('torch','numpy'))
            return real_import(name,*args,**kwargs)
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.top.rglob('*') if p.is_file()}
        with patch('builtins.__import__',side_effect=pure_import):answer=metrics.summarize(self.commit)
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.top.rglob('*') if p.is_file()}
        self.assertEqual(before,after);self.assertEqual(answer['evidence_writes'],0)


class PureMetrics(unittest.TestCase):
    def test_G5_distribution_math_and_integer_mixed_count(self):
        c=canonical_fixture();metrics.check_metrics(c)
        self.assertEqual(metrics.exact_count(c['prequential']['predicted_diversity']['actual_mixed_fraction'],448),224)
        c['K_distribution']['0']+=1
        with self.assertRaisesRegex(ValueError,'K0..5 distribution differs'):metrics.check_metrics(c)
        with self.assertRaisesRegex(ValueError,'exact integer count'):metrics.exact_count(.123456,64)
    def test_recent10_window_and_undefined_auc_never_fabricated_zero(self):
        c=canonical_fixture(undefined=True);metrics.check_metrics(c)
        outers=[dict(outer=i,prequential=c['prequential'],K_distribution=c['K_distribution']) for i in range(1,13)]
        summary=metrics.aggregate(outers[-10:]);self.assertEqual((summary['first_outer'],summary['last_outer']),(3,12))
        self.assertIsNone(summary['cohorts']['all']['std_mixed_auc_within_policy_outer_mean'])
        self.assertEqual(summary['cohorts']['all']['std_mixed_auc_defined_outers'],0)
        self.assertIsNone(summary['top_over_random_mixed_rate_ratio']);self.assertFalse(summary['ratio_defined'])

    def test_each_cohort_counts_must_partition_and_original_schema_rejects_explicit_None(self):
        value=canonical_fixture();value['prequential']['random_exploration']['all_wrong_fraction']=47/64
        with self.assertRaisesRegex(ValueError,'cohort mixed/allwrong/allcorrect'):metrics.check_metrics(value)
        value=canonical_fixture();value['prequential']['all']['std_mixed_auc']=None
        with self.assertRaisesRegex(ValueError,'within-policy rank statistic invalid'):metrics.check_metrics(value)


if __name__=='__main__':unittest.main()
