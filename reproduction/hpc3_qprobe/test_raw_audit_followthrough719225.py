"""Actual attempt semantics and fail-closed source/birth consumers; CPU stdlib."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
from reproduction.hpc3_qprobe import raw_audit_followthrough719225 as audit


class StartupCompletedPrefix(unittest.TestCase):
    def test_start_authenticates_only_actual_prefix_but_declares_future200(self):
        old_base=audit.BASE.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718509-20261014'
        old,new={'original':'718650'},{'original':'719225'}
        history=Mock(return_value=[dict(base=old_base,run=old,first=1,last=5),
                                   dict(base=audit.BASE,run=new,first=6,last=24)])
        def producer(base):
            if base==old_base:
                return old,dict(producer_job='718650',producer_commit='5816a20f331e10aaff9bbb95c313611ead289fd7'),'FAILED'
            return new,dict(producer_job='719225',producer_commit='1843babdd96b387dd10d91e3207c9ea9489b6941'),'RUNNING'
        with patch.dict(sys.modules,{'reproduction.hpc3_qprobe.verify_goal':SimpleNamespace(history_parts=history)}), \
             patch.object(audit.ctl,'read',return_value=new), \
             patch.object(audit,'rows',return_value=[dict(step=24)]), \
             patch.object(audit.ctl,'producer',side_effect=producer):
            parts=audit.plan()
        history.assert_called_once_with(audit.BASE,new,last=24)
        self.assertEqual([(p['first'],p['last']) for p in parts],[(1,5),(6,200)])


class TerminalPublicationReadiness(unittest.TestCase):
    def test_partial_terminal_and_active_ledger_wait_and_wrong_attempt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(audit.ctl,'OWNER',os.getuid()):
            base=Path(temporary).resolve();aid='a'*32
            task=dict(producer=dict(base=str(base)),attempt_id=aid,step=24,complete_outer=True)
            folder=base/'artifacts/question-probe/attempts'/aid
            terminal=base/'artifacts/question-probe/completed'/(aid+'.json')
            required=[folder/'real-rollout.pt',folder/'selection.json',folder/'rollout-audit.json',
                      base/'logs'/('behavior-probability-attempt-'+aid+'.json'),
                      base/'artifacts/behavior-traces'/('attempt-'+aid+'.json')]
            for path in required:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            terminal.parent.mkdir(parents=True,exist_ok=True)
            ledger=base/'logs/actual-updates.jsonl'
            ledger.write_text(json.dumps(dict(attempt_id=aid))+'\n')
            self.assertFalse(audit.task_ready(task))
            terminal.write_text('{')
            self.assertFalse(audit.task_ready(task))
            terminal.write_text(json.dumps(dict(attempt_id=aid,outer=24)))
            ledger.write_text(json.dumps(dict(attempt_id=aid)))
            self.assertFalse(audit.task_ready(task))
            ledger.write_text(json.dumps(dict(attempt_id=aid))+'\n')
            self.assertTrue(audit.task_ready(task))
            terminal.write_text(json.dumps(dict(attempt_id='b'*32,outer=24)))
            self.assertFalse(audit.task_ready(task))


class OriginalAttemptSequence(unittest.TestCase):
    def setUp(self):
        self.producer=dict(base='/original',run_id='run',producer_commit='commit',producer_job='719225',
                           source_manifest_sha256='manifest',native_allocation_sha256='allocation')
        self.part=dict(first=1,last=200,producer=self.producer)

    def row(self,index,outer,writes,completed):
        return dict(attempt_id=('%032x'%index),attempt_index=index,logical_update=outer,
                    run_id='run',slurm_job_id='719225',optimizer_writes_this_attempt=writes,
                    opportunities_processed=20,complete_outer=writes>0,scheduler_advances=int(writes>0),
                    completed_outer=completed,status='complete' if writes>0 else 'recoverable_guard')

    def test_partial_write_success_and_fully_skipped_attempt_both_retained(self):
        rows=[self.row(1,1,3,1),self.row(2,2,0,1),self.row(3,2,1,2)]
        tasks=audit.tasks_from_rows([self.part],[rows])
        self.assertEqual([t['complete_outer'] for t in tasks],[True,False,True])
        self.assertEqual([t['step'] for t in tasks],[1,2,2])
        self.assertEqual([t['actual_optimizer_writes'] for t in tasks],[3,0,1])

    def test_update201_missing_attempt_duplicate_or_false_scheduler_rejected(self):
        valid=self.row(1,1,20,1)
        cases=([dict(valid,logical_update=201)], [dict(valid,attempt_index=2)],
               [valid,dict(self.row(2,2,20,2),attempt_id=valid['attempt_id'])],
               [dict(valid,scheduler_advances=0)])
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                audit.tasks_from_rows([self.part],[rows])

    def test_original_source_job_is_not_replaced_by_current_job(self):
        old=dict(self.producer,producer_job='718650')
        with self.assertRaises(ValueError):
            audit.tasks_from_rows([dict(self.part,producer=old)],[[self.row(1,1,20,1)]])

    def test_copied_current_prefix_cannot_be_hidden(self):
        current=dict(self.part,first=6)
        with self.assertRaises(ValueError):
            audit.tasks_from_rows([current],[[self.row(1,1,20,1)]])


class WaitedResultConsumer(unittest.TestCase):
    def setUp(self):
        p=OriginalAttemptSequence();p.setUp()
        self.task=audit.tasks_from_rows([p.part],[[p.row(1,1,3,1)]])[0]
        self.supervisor=dict(pid=100,start_ticks=1000,sid=100,uid=14090)
        child=dict(pid=101,start_ticks=1001,sid=100,uid=14090)
        self.result=dict(task=self.task,process=self.supervisor,child=child,actual_child_exit_code=0,
                         stderr=dict(bytes=0))
        self.output=dict(status='ACTUAL_RAW_CANONICAL_TRACE_ALLTOKENS_AND_ACTOR_ROWS_EQUAL',step=1,
                         job='719225',source_commit='commit',source_manifest=dict(sha256='manifest'),
                         allocation_file=dict(sha256='allocation'),actual_attempt_transaction=dict(
                         attempt_id=self.task['attempt_id'],attempt_index=1,complete_outer=True,
                         actual_optimizer_writes=3,attempt_record_sha256=self.task['record_sha256']),
                         canonical_proof=dict(actual_head_committed=True,actual_raw_decode_strict_terminal_em_equal=True,
                         actual_512_qid_text_gold_equal=True,actual_selection_G5_E64_top448_prequential_head_equal=True,
                         questions=512,trajectories=2560),hf_forward_performed=False,model_weights_loaded=False,
                         generated_trajectories=0,actor_or_head_updates_performed=0,evidence_files_written=0)
        for key in ('actual_private_trace_byte_equal','actual_all2560_ordered_ids_equal','actual_512_parent_digests_equal',
                    'actual_2560_actor_tensor_rows_equal','actual_all_retained_token_contexts_equal',
                    'actual_selected_logprob_projection_equal'):self.output[key]=True

    def test_complete_samefd_raw_and_wait_proof_accepted(self):
        audit.accept_result(self.task,self.result,self.output,self.supervisor,False)

    def test_header_missing_birth_live_child_wrong_source_or_false_raw_labels_rejected(self):
        for key,change in (('child',None),('actual_child_exit_code',1),('stderr',dict(bytes=1))):
            result=dict(self.result,**{key:change})
            with self.subTest(field=key),self.assertRaises(ValueError):
                audit.accept_result(self.task,result,self.output,self.supervisor,False)
        with self.assertRaises(ValueError):
            audit.accept_result(self.task,self.result,self.output,self.supervisor,True)
        for key,value in (('job','718650'),('source_manifest',dict(sha256='changed'))):
            with self.subTest(field=key),self.assertRaises(ValueError):
                audit.accept_result(self.task,self.result,dict(self.output,**{key:value}),self.supervisor,False)
        output=copy.deepcopy(self.output);output['canonical_proof']['actual_raw_decode_strict_terminal_em_equal']=False
        with self.assertRaises(ValueError):
            audit.accept_result(self.task,self.result,output,self.supervisor,False)


if __name__=='__main__':unittest.main()
