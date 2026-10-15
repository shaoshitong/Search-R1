"""CPU contract/receipt rejection tests, never native training evidence."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.common8.formal_contract import (
    CONTRACT_KIND, FORMAL_COUNTING, FORMAL_MATHEMATICS, FORMAL_SAFEGUARDS, SAFETY_MODE, PROBE_ROUNDS, SAVE_STEPS,
    assert_formal_run, consumed_lr, verify_formal_config,
)
from reproduction.common8.policy_calibration_plan import MATHEMATICS, canonical_bytes
from reproduction.common8.runtime import budget_rows, refresh_after_checkpoint, verify_scheduler_receipts
from reproduction.common8 import test_configuration


class FormalContractTests(unittest.TestCase):
    def test_new_contract_keeps_original_diagnostic_math_immutable(self):
        self.assertEqual(MATHEMATICS['target_actual_outer_updates'],300)
        self.assertEqual(MATHEMATICS['learning_rate_schedule'],'constant')
        self.assertEqual(MATHEMATICS['reference_kl_coefficient'],.001)
        self.assertEqual(FORMAL_MATHEMATICS['target_actual_outer_updates'],200)
        self.assertEqual(FORMAL_MATHEMATICS['lr_warmup_steps'],57)
        self.assertEqual(FORMAL_MATHEMATICS['entropy_coefficient'],0.)
        self.assertEqual(FORMAL_MATHEMATICS['reference_kl_coefficient'],0.)
        self.assertNotIn('optimizer_completion_rule',MATHEMATICS)
        self.assertEqual(FORMAL_MATHEMATICS['optimizer_completion_rule'],FORMAL_COUNTING)
        self.assertEqual(FORMAL_MATHEMATICS['optimizer_opportunities_per_outer'],20)
        self.assertEqual(FORMAL_MATHEMATICS['policy_safety_mode'],SAFETY_MODE)
        self.assertEqual(FORMAL_SAFEGUARDS['target_old_policy_kl'],.2)
        self.assertEqual(PROBE_ROUNDS,(0,50,100,150))
        self.assertEqual(SAVE_STEPS,(5,50,100,150,200))

    def test_exact_consumed_lr_boundaries_and_bad_outer_rejection(self):
        for step,value in ((1,0.),(2,1e-6*(1/57)),(57,1e-6*(56/57)),(58,1e-6),(200,1e-6)):
            self.assertEqual(consumed_lr(step),value)
        for step in (0,201,True,1.0):
            with self.subTest(step=step),self.assertRaises(AssertionError):consumed_lr(step)

    def test_formal_config_rejects_horizon_objective_warmup_micro_and_old_scope(self):
        from reproduction.common8.configuration import resolved
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as root:
            base,run=test_configuration.ConfigurationTests().fixture(root)
            _,config,_=resolved(base)
            document=OmegaConf.to_container(config,resolve=True)
            proof=verify_formal_config(document,run,base)
            self.assertEqual(proof['contract_kind'],CONTRACT_KIND)
            for path,value in ((('trainer','total_training_steps'),301),
                    (('actor_rollout_ref','actor','optim','lr_warmup_steps'),0),
                    (('actor_rollout_ref','actor','optim','lr_warmup_steps_ratio'),57/200),
                    (('actor_rollout_ref','actor','entropy_coeff'),.001),
                    (('actor_rollout_ref','actor','kl_loss_coef'),.001),
                    (('actor_rollout_ref','actor','use_kl_loss'),False),
                    (('actor_rollout_ref','actor','ppo_micro_batch_size'),128),
                    (('actor_rollout_ref','actor','policy_safety','mode'),'legacy'),
                    (('actor_rollout_ref','actor','policy_safety','target_old_policy_kl'),.01),
                    (('actor_rollout_ref','rollout','temperature'),.5)):
                bad=copy.deepcopy(document);part=bad
                for key in path[:-1]:part=part[key]
                part[path[-1]]=value
                with self.subTest(path=path),self.assertRaises(AssertionError):verify_formal_config(bad,run,base)
            for key,value in (('contract_kind','constant300'),('target_updates',300),('completed_updates',1),
                              ('policy_safety_mode','legacy')):
                bad=copy.deepcopy(run);bad[key]=value
                with self.subTest(key=key),self.assertRaises(AssertionError):assert_formal_run(bad)

    def scheduler_fixture(self,root,last=2,successes=None,include_abort=False):
        base,run=test_configuration.ConfigurationTests().fixture(root)
        run.update(source_commit='a'*40)
        source=hashlib.sha256(canonical_bytes(run['source_sha256'])).hexdigest()
        contract=hashlib.sha256(canonical_bytes(run['budget'])).hexdigest()
        attempts=[];records=[];physical=0
        if include_abort:
            attempts.append(dict(step=1,processed=5,successes=3,complete=False))
        attempts += [dict(step=i,processed=20,successes=(successes[i-1] if successes else 20),complete=True)
                     for i in range(1,last+1)]
        gates={}
        for ordinal,attempt in enumerate(attempts,1):
            attempt.update(attempt_id=f'{ordinal:032x}',attempt_index=ordinal,before=physical,
                           proof_sha256=f'{ordinal:064x}')
            physical+=attempt['successes'];attempt['after']=physical
            if attempt['complete']:
                gates[str(attempt['step'])]=dict(proof_sha256=attempt['proof_sha256'])
                records.append(dict(step=attempt['step'],attempt_id=attempt['attempt_id'],attempt_index=ordinal,
                    optimizer_writes_before=attempt['before'],optimizer_writes_after=physical,opportunities_processed=20))
        (base/'logs/actual-updates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
        (base/'logs/training-launch-claim.json').write_text(json.dumps(dict(run_id=run['run_id'],slurm_job_id='123',
            shell_pid=10,shell_start_ticks=999)))
        # Synthetic allocation required by the current durable scheduler reader;
        # this fixture does not assert that any CUDA allocation was performed.
        (base/'logs/native-allocation-admission.json').write_text(json.dumps(dict(
            accepted=True,run_id=run['run_id'],slurm_job_id='123',
            fixture_scope='constructed CPU protocol evidence only')))
        leader=dict(pid=10,start_ticks=999,sid=10,uid=123)
        ledger={str(100+rank):dict(pid=100+rank,start_ticks=1000+rank,sid=10,uid=123) for rank in range(8)}
        (base/'logs/process-ledger.json').write_text(json.dumps(dict(run_id=run['run_id'],
            jobs={'trainer':dict(leader=leader,ledger=ledger,state='STARTED')})))
        (base/'logs/run-manifest.json').write_text(json.dumps(run))
        for rank in range(8):
            rows=[];workers=[]
            for attempt in attempts:
                step=attempt['step'];ordinal=attempt['attempt_index'];before=attempt['before']
                cumulative=before
                for mini in range(attempt['processed']):
                    success=mini<attempt['successes']
                    context=dict(logical_update=step,attempt_id=attempt['attempt_id'],attempt_index=ordinal,minibatch_index=mini)
                    workers.append(dict(run_id=run['run_id'],rank=rank,kind='OPPORTUNITY',context=context,
                        opportunity_index=mini,processed=True,optimizer_write_succeeded=success,
                        status='written' if success else 'skipped_guard',optimizer_writes_before=cumulative,
                        optimizer_writes_after=cumulative+int(success)))
                    if success:
                        cumulative+=1
                        workers.append(dict(run_id=run['run_id'],rank=rank,kind='OPTIMIZER_WRITE',context=context,
                                            write_count=cumulative,learning_rates=[consumed_lr(step)]))
                kinds=('BEGIN','COMMIT_PREPARED','COMPLETE') if attempt['complete'] else ('BEGIN','ABORTED_ATTEMPT')
                for stage,kind in enumerate(kinds):
                    after=step if kind=='COMPLETE' else step-1
                    written=kind!='BEGIN'
                    rows.append(dict(schema_version=1,kind=kind,role='fresh200_outer_scheduler',
                        contract_kind=CONTRACT_KIND,run_id=run['run_id'],source_commit=run['source_commit'],
                        source_map_sha256=source,scheduler_contract_sha256=contract,slurm_job_id='123',rank=rank,world_size=8,
                        logical_update=step,warmup_steps=57,base_learning_rate=1e-6,
                        process_pid=100+rank,process_start_ticks=1000+rank,
                        observed_unix=100.+ordinal*4+stage*.1,
                        attempt_id=attempt['attempt_id'],attempt_index=ordinal,
                        batch_gate_sha256=attempt['proof_sha256'],
                        completed_outer_before=step-1,completed_outer_after=after,
                        scheduler_last_epoch_before=step-1,scheduler_last_epoch_after=after,
                        scheduler_step_count_before=step,scheduler_step_count_after=after+1,
                        writes_before=before,writes_after=attempt['after'] if written else before,
                        actual_optimizer_writes=attempt['successes'] if written else 0,
                        opportunities_processed=attempt['processed'] if written else 0,optimizer_opportunities=20,
                        recoverable_guard=True,retained_partial_adam_writes=True,no_automatic_rollback=True,
                        scheduler_unchanged_on_rejection=True,scheduler_advances=int(kind=='COMPLETE'),
                        consumed_learning_rates=[consumed_lr(step)],next_learning_rates=[1e-6*min(after/57,1.)]))
            (base/'logs'/f'scheduler-rank-{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (base/'logs'/f'optimizer-rank-{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in workers))
        return base,run,gates

    def verify_scheduler(self,base,run,first,last,gates):
        # This test isolates the durable scheduler/opportunity reader. The
        # full2560 native trace gate has its own real-byte rejection tests.
        def gate(_base,_run,step,update_record=None,attempt_id=None,attempt_index=None):
            ordinal=update_record['attempt_index'] if update_record is not None else attempt_index
            return dict(proof_sha256=f'{ordinal:064x}')
        with patch('reproduction.common8.behavior_runtime.read_gate_receipt',side_effect=gate):
            return verify_scheduler_receipts(base,run,first,last,gates)

    def test_durable_scheduler_binding_rejects_duplicate_rejected_wrong_counter_or_lr(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root)
            proof=self.verify_scheduler(base,run,1,2,gates)
            self.assertEqual(len(proof),8)
            path=base/'logs/scheduler-rank-3.jsonl';original=path.read_bytes()
            for field,value in (('kind','REJECTED'),('source_commit','b'*40),
                    ('scheduler_step_count_after',20),('scheduler_last_epoch_after',20),
                    ('actual_optimizer_writes',19),('writes_after',19),
                    ('consumed_learning_rates',[1e-6]),('batch_gate_sha256','wrong'),
                    ('process_pid',999),('contract_kind','old')):
                rows=[json.loads(line) for line in original.splitlines()];rows[1][field]=value
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.subTest(field=field),self.assertRaises(AssertionError):self.verify_scheduler(base,run,1,2,gates)
            path.write_bytes(original+original.splitlines()[0]+b'\n')
            with self.assertRaises(AssertionError):self.verify_scheduler(base,run,1,2,gates)

    def test_scheduler_ranks_share_the_exact_ledger_snapshot_used_for_births(self):
        from reproduction.common8.ledger_snapshot import read_owned_ledger_snapshot
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root)
            ledger=base/'logs/process-ledger.json'
            original=ledger.read_bytes()

            def snapshot_then_publish_later(path):
                document,proof=read_owned_ledger_snapshot(path)
                later=copy.deepcopy(document)
                later['jobs']['trainer']['ledger']={}
                temp=ledger.with_suffix('.replacement')
                temp.write_text(json.dumps(later));os.replace(temp,ledger)
                return document,proof

            with patch('reproduction.common8.runtime.read_owned_ledger_snapshot',
                       side_effect=snapshot_then_publish_later) as snapshot:
                proof=self.verify_scheduler(base,run,1,2,gates)
            self.assertEqual(snapshot.call_count,1)
            self.assertEqual({r['owned_process_ledger_sha256'] for r in proof.values()},
                             {hashlib.sha256(original).hexdigest()})
            self.assertNotEqual(hashlib.sha256(ledger.read_bytes()).hexdigest(),
                                hashlib.sha256(original).hexdigest())
            # A genuinely missing birth in the actual next snapshot stays fatal.
            with self.assertRaisesRegex(AssertionError,'birth absent'):
                self.verify_scheduler(base,run,1,2,gates)

    def test_fatal_console_invalidates_otherwise_complete_scheduler_triplets(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root)
            self.assertEqual(len(self.verify_scheduler(base,run,1,2,gates)),8)
            Path(run['train_log']).write_text('SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='+json.dumps(dict(
                run_id=run['run_id'],actual_scheduler_last_epoch=2,
                actual_scheduler_step_count=3,writes_after=40,fatal_no_automatic_resume=True))+'\n')
            with self.assertRaisesRegex(AssertionError,'Fatal scheduler console'):
                self.verify_scheduler(base,run,1,2,gates)

    def test_first_outer_twenty_actual_zero_lr_writes_not_constant_lr(self):
        from reproduction.common8.test_contract import evidence
        lines=evidence();self.assertEqual(len(budget_rows('\n'.join(lines),1,1)),8)
        changed=lines[0].replace('[0.0]','[1e-06]')
        with self.assertRaises(AssertionError):budget_rows('\n'.join([changed]+lines[1:]),1,1)

    def test_scheduler_births_must_match_independent_owned_ledger_and_distinct_ranks(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root)
            for mode in ('unknown','shared'):
                path=base/'logs/scheduler-rank-3.jsonl';original=path.read_bytes()
                rows=[json.loads(line) for line in original.splitlines()]
                for row in rows:
                    row['process_pid']=999 if mode=='unknown' else 100
                    row['process_start_ticks']=9999 if mode=='unknown' else 1000
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.subTest(mode=mode),self.assertRaises(AssertionError):
                    self.verify_scheduler(base,run,1,2,gates)
                path.write_bytes(original)

    def test_actual_five_scheduler_interval_matches_full_two_hundred_receipts(self):
        from reproduction.common8.verify_full import verify_five_scheduler_prefix
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root,last=200)
            five=self.verify_scheduler(base,run,1,5,gates)
            full=self.verify_scheduler(base,run,1,200,gates)
            for rank in range(8):
                selected=verify_five_scheduler_prefix(five[str(rank)],full[str(rank)])
                self.assertEqual(len(selected),15)
                bad=copy.deepcopy(five[str(rank)]);bad['selected_rows']=bad['selected_rows'][:10]
                with self.assertRaises(AssertionError):verify_five_scheduler_prefix(bad,full[str(rank)])

    def test_terminal200_never_starts_unused_probe_calibration(self):
        with patch('subprocess.run') as subprocess_run,patch('requests.Session') as session:
            self.assertIsNone(refresh_after_checkpoint(200,'unused'))
        subprocess_run.assert_not_called();session.assert_not_called()

    def test_opportunities_complete_with_fewer_writes_and_keep_abandoned_partial_writes(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root,successes=[17,1],include_abort=True)
            proof=self.verify_scheduler(base,run,1,2,gates)
            for receipt in proof.values():
                self.assertEqual(receipt['cumulative_actual_optimizer_writes'],21)
                self.assertEqual(receipt['complete_outer_optimizer_opportunities'],40)
            path=base/'logs/optimizer-rank-3.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows=[row for row in rows if not (row['kind']=='OPPORTUNITY' and row['context']['attempt_index']==2 and row['opportunity_index']==19)]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(AssertionError,'opportunity receipt'):
                self.verify_scheduler(base,run,1,2,gates)

    def test_first_five_prefix_keeps_aborted_attempts_without_counting_them_as_updates(self):
        from reproduction.common8.verify_full import verify_five_scheduler_prefix
        with tempfile.TemporaryDirectory() as root:
            base,run,gates=self.scheduler_fixture(root,last=6,include_abort=True)
            five=self.verify_scheduler(base,run,1,5,gates)
            full=self.verify_scheduler(base,run,1,6,gates)
            selected=verify_five_scheduler_prefix(five['0'],full['0'])
            self.assertEqual(len(selected),17)
            self.assertEqual(sum(row['kind']=='COMPLETE' for row in selected),5)
            self.assertEqual(sum(row['kind']=='ABORTED_ATTEMPT' for row in selected),1)
            bad=copy.deepcopy(five['0']);bad['selected_rows']=bad['selected_rows'][2:]
            with self.assertRaises(AssertionError):verify_five_scheduler_prefix(bad,full['0'])


if __name__=='__main__':unittest.main()
