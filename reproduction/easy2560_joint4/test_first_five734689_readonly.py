"""Actual6..8 scalar schemas and small synthetic6..10 disposition fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FOLDER=Path(__file__).parent/'code_backup/micro8-after731063-20261018'
SPEC=importlib.util.spec_from_file_location('first_five734689',FOLDER/'accept_first_five734689_readonly.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class FirstFiveReadonlyTests(unittest.TestCase):
    def actual(self):return json.loads((FOLDER/'actual-native-terminal6_8-fixture.json').read_text())['rows']

    def fixture(self, *, last=10, skip=False, rollback=False):
        template=self.actual()[0]
        run=dict(run_id=m.RUN,source_commit=m.COMMIT,training_world_size=4,actor_micro_batch_per_rank=8,
            actor_logical_micro_batch_per_rank=4,all_correct_virtual_zero=False,source_sha256={'fixture.py':'a'*64},
            budget=dict(ppo_mini_batch_size=128,ppo_micro_batch_size=32,per_rank_actor_logical_micro_batch=4),
            full_state_restore=dict(physical_micro_transition=dict(source_physical_micro_per_rank=16,
                target_physical_micro_per_rank=8,target_accumulation_steps=4,global_mini_batch=128,logical_micro_per_rank=4)))
        d=dict(run=run,restore_chain=dict(current_job=m.JOB,current_source_commit=m.COMMIT,source_job='730226',
            completed_outer=5,historical_optimizer_writes=100,historical_attempt_index=5,
            native4_worker_driver_restore_proofs_authenticated=True),updates=[],attempts=[],
            optimizers={str(r):[] for r in range(4)},schedulers={str(r):[] for r in range(4)},
            restored_workers={str(r):dict(process_identity=dict(pid=100+r,start_ticks=1000+r)) for r in range(4)},
            next11=dict(kind='native_candidate_progress',value=dict(step=11,candidates=512,mixed=170)),
            pipeline_stage='GRPO_1_TO_200',error_lines=[])
        before=100
        for step in range(6,last+1):
            statuses=['written']*20
            if step==7 and (skip or rollback):statuses[-1]='rolled_back_guard' if rollback else 'skipped_guard'
            writes=statuses.count('written');after=before+writes;aid=f'{step:032x}'
            u=dict(template['update'],step=step,attempt_id=aid,attempt_index=step,optimizer_writes_before=before,
                optimizer_writes_after=after,actual_optimizer_writes=writes,opportunities_processed=20,
                full_budget_complete=writes==20,update_status='complete' if writes==20 else 'partial')
            d['updates'].append(u)
            d['attempts'].append(dict(run_id=m.RUN,slurm_job_id=m.JOB,logical_update=step,attempt_index=step,attempt_id=aid,
                optimizer_writes_before=before,optimizer_writes_after=after,optimizer_writes_this_attempt=writes,
                opportunities_processed=20,opportunity_statuses=statuses,complete_outer=True,scheduler_advances=1,
                completed_outer=step,all_ranks_agreed=True,real_rank_count=4,opportunities_unattempted=0,
                full_budget_complete=writes==20,optimizer_rollbacks_this_attempt=int(rollback and step==7),
                optimizer_calls_this_attempt=writes+int(rollback and step==7)))
            for r in range(4):
                rows=d['optimizers'][str(r)];c=dict(attempt_id=aid,attempt_index=step,logical_update=step)
                cumulative=before
                for i,status in enumerate(statuses):
                    success=status=='written';rolled=status=='rolled_back_guard'
                    rows.append(dict(kind='OPPORTUNITY',run_id=m.RUN,rank=r,context=dict(c,minibatch_index=i),
                        opportunity_index=i,processed=True,status=status,optimizer_write_succeeded=success,
                        optimizer_write_rolled_back=rolled,optimizer_call_attempted=success or rolled,
                        optimizer_writes_before=cumulative,optimizer_writes_after=cumulative+int(success)))
                    cumulative+=int(success)
                    if success:rows.append(dict(kind='OPTIMIZER_WRITE',run_id=m.RUN,rank=r,
                        context=dict(c,minibatch_index=i),write_count=cumulative,learning_rates=[1e-6*(step-1)/57]))
                b=dict(template['ranks'][str(r)]['budget'],logical_update=step,attempt_id=aid,attempt_index=step,
                    optimizer_writes_before=before,optimizer_writes_after=after,cumulative_writes=after,
                    optimizer_writes=writes,full_budget_complete=writes==20)
                rows.append(b)
                s=dict(template['ranks'][str(r)]['schedule'],logical_update=step,attempt_id=aid,attempt_index=step,
                    completed_outer_before=step-1,completed_outer_after=step,scheduler_last_epoch_before=step-1,
                    scheduler_last_epoch_after=step,scheduler_step_count_before=step,scheduler_step_count_after=step+1,
                    writes_before=before,writes_after=after,actual_optimizer_writes=writes,
                    source_map_sha256=m.digest(run['source_sha256']),scheduler_contract_sha256=m.digest(run['budget']),
                    process_pid=100+r,process_start_ticks=1000+r,
                    consumed_learning_rates=[1e-6*(step-1)/57],next_learning_rates=[1e-6*step/57])
                d['schedulers'][str(r)]+=[dict(s,kind='BEGIN'),dict(s,kind='COMMIT_PREPARED'),s]
            before=after
        return d

    def test_all_four_actual_native_terminal_schemas6_7_8(self):
        for row in self.actual():
            for rank,native in row['ranks'].items():
                result=m.terminal(row['update'],native['budget'],native['schedule'],int(rank))
                self.assertEqual(result['processed'],20);self.assertEqual(result['physical_micro'],8)

    def test_complete_synthetic_five_and_positive_native11(self):
        result=m.inspect(self.fixture());self.assertTrue(result['accepted']);self.assertEqual(len(result['outers']),5)
        self.assertEqual(result['cumulative_retained_Adam_writes_per_rank'],200)

    def test_legal_processed20_skip_or_rollback_preserves_actual_counts(self):
        for rollback in (False,True):
            result=m.inspect(self.fixture(skip=not rollback,rollback=rollback))
            self.assertTrue(result['accepted']);self.assertEqual(result['cumulative_retained_Adam_writes_per_rank'],199)
            row=result['outers'][1];self.assertEqual((row['writes'],row['skips'],row['rollbacks']),(19,int(not rollback),int(rollback)))

    def test_not_yet10_or_epoch_only11_is_pending(self):
        self.assertFalse(m.inspect(self.fixture(last=8))['accepted'])
        d=self.fixture();d['next11']=None;self.assertEqual(m.inspect(d)['status'],'PENDING_FIRST_FIVE_OR_NATIVE11')
        d['next11']=dict(kind='native_epoch_header',line='epoch 0, step 11');self.assertFalse(m.inspect(d)['accepted'])

    def test_missing_native_terminal_is_pending_and_wrong_geometry_or_write_fails(self):
        d=self.fixture();d['schedulers']['3'].pop();self.assertFalse(m.inspect(d)['accepted'])
        for change in ('geometry','write','source','rollback'):
            d=self.fixture()
            if change=='geometry':d['optimizers']['0'][-1]['local_micro']=16
            elif change=='write':d['updates'][-1]['actual_optimizer_writes']=19
            elif change=='source':d['run']['source_commit']='a'*40
            else:d['attempts'][0]['optimizer_rollbacks_this_attempt']=1
            with self.subTest(change=change),self.assertRaises(ValueError):m.inspect(d)

    def test_json_without_final_newline_and_later_partial_jsonl_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'small.json';path.write_text('{"run":1}')
            self.assertEqual(json.loads(m.read_prefix(path)[0]),{'run':1})
            path.write_bytes(b'{"step":10}\n{"step":11}\n{"partial":')
            rows,proof=m.read_prefix(path,jsonl=True);self.assertEqual(rows,[{'step':10}]);self.assertTrue(proof['prefix_only'])


if __name__=='__main__':unittest.main()
