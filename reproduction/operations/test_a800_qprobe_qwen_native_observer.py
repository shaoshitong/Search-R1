"""One metadata-only CPU fixture of reused fresh counters and native prefixes."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import unittest

from reproduction.operations import a800_qprobe_qwen_native_observer as observer


class NativeObserverTests(unittest.TestCase):
    def namespace(self):
        program=observer.remote_program();nodes=ast.parse(program).body
        functions=[node for node in nodes if isinstance(node,ast.FunctionDef)]
        namespace=dict(Unknown=type('Unknown',(Exception,),{}),math=math,re=re,hashlib=hashlib,json=json,Path=Path,
            SMALL_LIMIT=2*1024**2,LEDGER_LIMIT=8*1024**2,RANK_LEDGER_LIMIT=32*1024**2,PROCESS_LEDGER_LIMIT=128*1024**2,
            payload=dict(after_step=0),run=dict(run_id='1'*32,execution_id='2'*32,source_commit='3'*40,
                execution_backend='ssh_local',slurm_job_id=None),driver_attempt_partial=False,
            complete_updates=None,closed_attempts=None)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'<published-metadata-fixture>','exec'),namespace)
        return namespace

    def fixture(self):
        run='1'*32;aid='4'*32;gate='a'*64;statuses=['written']*18+['skipped_guard']*2
        attempt=dict(run_id=run,execution_id='2'*32,execution_backend='ssh_local',slurm_job_id=None,source_commit='3'*40,
            attempt_id=aid,attempt_index=1,logical_update=1,opportunities_processed=20,optimizer_writes_this_attempt=18,
            opportunity_statuses=statuses,optimizer_writes_before=0,optimizer_writes_after=18,recovery_protocol_version=2,
            complete_outer=True,status='complete',all_ranks_agreed=True,real_rank_count=4,scheduler_advances=1,completed_outer=1,
            full_budget_complete=False,opportunities_unattempted=0,retained_partial_adam_writes=False,
            optimizer_calls_this_attempt=18,optimizer_rollbacks_this_attempt=0,current_step_rollback=True,no_automatic_rollback=False,
            cumulative_aborted_attempts=0,cumulative_full_budget_outers=0)
        update=dict(run_id=run,step=1,attempt_id=aid,attempt_index=1,optimizer_writes_before=0,optimizer_writes_after=18,
            actual_optimizer_writes=18,opportunities_processed=20,recovery_protocol_version=2,full_budget_complete=False,
            opportunities_unattempted=0,behavior_probability_gate_sha256=gate,behavior_trace_sha256='b'*64,
            metrics={'critic/rewards/mean':.3,'actor/update_opportunities_this_attempt':20,
                     'actor/optimizer_steps_this_update':18,'actor/skipped_update_opportunities':2})
        logs={'logs/actual-attempts.jsonl':[attempt],'logs/actual-updates.jsonl':[update]}
        for rank in range(4):
            records=[];writes=0
            for index,status in enumerate(statuses):
                wrote=status=='written'
                records.append(dict(run_id=run,rank=rank,kind='OPPORTUNITY',recovery_protocol_version=2,
                    context=dict(attempt_id=aid,attempt_index=1,logical_update=1,behavior_probability_gate_sha256=gate,
                                 minibatch_index=index,prior_optimizer_writes=writes),opportunity_index=index,
                    optimizer_writes_before=writes,optimizer_writes_after=writes+int(wrote),processed=True,
                    optimizer_write_succeeded=wrote,status=status,optimizer_write_rolled_back=False,optimizer_call_attempted=wrote))
                writes+=int(wrote)
            records.append(dict(run_id=run,rank=rank,kind='OUTER_BUDGET',attempt_id=aid,attempt_index=1,logical_update=1,
                recovery_protocol_version=2,opportunities_processed=20,optimizer_writes=18,cumulative_writes=18,
                optimizer_writes_before=0,optimizer_writes_after=18,full_budget_complete=False))
            logs['logs/optimizer-rank-'+str(rank)+'.jsonl']=records
        return logs

    def test_fresh_four_rank_twenty_rule_native_prefix_and_unknown_partial(self):
        for path,_,sha in observer.READERS:
            self.assertEqual(hashlib.sha256((observer.OPS/path).read_bytes()).hexdigest(),sha)
        env=self.namespace();logs=self.fixture();partial={'rank':False}
        def rows(relative,**kwargs):
            return copy.deepcopy(logs[relative]),dict(scan_from_start=True,complete_prefix_sha256='a'*64,
                partial_final_status='unknown_not_counted' if partial['rank'] and 'rank-3' in relative else None,
                partial_final_bytes=int(partial['rank'] and 'rank-3' in relative))
        env['rows']=rows
        reward=env['reward_window']();counts=env['opportunity_counts']()
        self.assertEqual(reward['newly_complete_steps'],[1]);self.assertEqual(reward['window_mean'],.3)
        self.assertEqual((counts['current_execution_processed'],counts['current_execution_successful_adam'],counts['current_execution_skip']),(20,18,2))
        self.assertEqual(counts['current_execution_full_budget_complete_outers'],0)
        self.assertTrue(counts['four_rank_complete_prefix_agreement'])
        self.assertEqual(len(counts['per_rank']),4)
        partial['rank']=True
        unknown=env['opportunity_counts']()
        self.assertEqual(unknown['status'],'unknown')
        self.assertIsNone(unknown['current_execution_successful_adam'])
        self.assertIsNone(unknown['current_execution_processed'])
        env['payload']['after_step']=1
        self.assertIsNone(env['reward_window']()['window_mean'])
        env['payload']['after_step']=2
        self.assertEqual(env['section'](env['reward_window'])['status'],'unknown')
        self.assertEqual(env['native_adam'](0),200);self.assertEqual(env['native_adam'](1),204)
        self.assertEqual(env['native_adam'](8),344);self.assertEqual(env['native_adam'](9),376)
        self.assertEqual(env['native_version']('qprobe-qwen:1:1:'+'a'*16,1),'qprobe-qwen:1:1:'+'a'*16)
        with self.assertRaises(env['Unknown']):env['native_version']('qprobe:1:1:'+'a'*16,1)
        self.assertNotIn('torch',observer.__dict__)


if __name__=='__main__':unittest.main()
