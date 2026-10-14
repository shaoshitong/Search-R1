"""CPU consumer tests: original bounded history stays in its own producer."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.common8.verify_full import qprobe_restoration_kind
from reproduction.hpc3_qprobe import verify_goal as goal


SAVED_KIND='question_probe_saved_checkpoint'


class HistoryFixture:
    def __init__(self,root):
        self.old=Path(root).resolve()/'original';self.new=Path(root).resolve()/'resumed'
        science=dict(run_id='c'*32,training_world_size=4,input_sha256={'train.parquet':'t'},
            initial_model_sha256={'model':'m'},budget={'world_size':4},
            question_probe=dict(candidate_questions=5120,selected_questions=512,
                responses_per_question=5,exploration_questions=64,seed=20261013))
        self.parent=dict(science,source_commit='1'*40,source_sha256={'worker':'old'})
        self.run=dict(science,source_commit='2'*40,source_sha256={'worker':'new'},
            full_state_restore=dict(kind=SAVED_KIND,completed_outer=5))
        self.parts=[dict(base=self.old,run=self.parent,first=1,last=5),
                    dict(base=self.new,run=self.run,first=6,last=200)]
        for part in self.parts:
            base=part['base'];(base/'logs').mkdir(parents=True)
            (base/'logs/run-manifest.json').write_text(json.dumps(part['run']))
            rows=[dict(run_id='c'*32,step=i,origin=str(base)) for i in range(part['first'],part['last']+1)]
            self.lines(base/'logs/actual-updates.jsonl',rows)

    @staticmethod
    def lines(path,rows):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def provider(self,parts=None):
        module=ModuleType('search_g1.qprobe_saved_checkpoint');module.SAVED_KIND=SAVED_KIND
        module.history_parts=lambda base,run,last=200:copy.deepcopy(self.parts if parts is None else parts)
        return patch.dict(sys.modules,{'search_g1.qprobe_saved_checkpoint':module})

    def attempts(self):
        for part in self.parts:
            base=part['base'];root=base/'artifacts/question-probe'
            (root/'completed').mkdir(parents=True);rows=[]
            for i in range(part['first'],part['last']+1):
                aid=f'{i:032x}';directory=root/'attempts'/aid;directory.mkdir(parents=True)
                row=dict(attempt_id=aid,attempt_index=i,logical_update=i,complete_outer=True,
                    completed_outer=i,run_id='c'*32);rows.append(row)
                selection=dict(step=i,attempt_id=aid,policy_snapshot_id='snapshot-'+str(i),head_version='h'+str(i-1))
                rollout=dict(step=i,attempt_id=aid,policy_snapshot_id=selection['policy_snapshot_id'])
                (directory/'selection.json').write_text(json.dumps(selection))
                (directory/'rollout-audit.json').write_text(json.dumps(rollout))
                event=dict(outer=i,successful_actor_snapshot_id='c'*32+':'+str(i),
                    prediction_version=selection['head_version'],label_policy_snapshot_id=selection['policy_snapshot_id'],
                    head_update={'update_count':i},prequential={'measured':True})
                (root/'completed'/(aid+'.json')).write_text(json.dumps(event))
            self.lines(base/'logs/actual-attempts.jsonl',rows)
            for rank in range(4):
                self.lines(base/'logs'/f'scheduler-rank-{rank}.jsonl',[dict(logical_update=part['first'])])
                self.lines(base/'logs'/f'optimizer-rank-{rank}.jsonl',[dict(context={'logical_update':part['first']})])


class GoalHistoryTests(unittest.TestCase):
    def test_independent_saved_kind_keeps_pair_and_legacy_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory)
            with fixture.provider():
                self.assertEqual(qprobe_restoration_kind({'kind':SAVED_KIND}),SAVED_KIND)
                self.assertEqual(qprobe_restoration_kind({'kind':'question_probe_pair'}),'question_probe_pair')
                self.assertIsNone(qprobe_restoration_kind({'kind':'legacy'}))
                self.assertIsNone(qprobe_restoration_kind(None))

    def test_actual_two_producer_raw_ranges_preserve_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory)
            with fixture.provider():parts=goal.history_parts(fixture.new,fixture.run)
            rows,proofs=goal.part_updates(parts)
            self.assertEqual([r['step'] for r in rows],list(range(1,201)))
            self.assertEqual(rows[4]['origin'],str(fixture.old));self.assertEqual(rows[5]['origin'],str(fixture.new))
            self.assertEqual([r['source_commit'] for r in proofs],['1'*40,'2'*40])
            self.assertEqual(goal.step_part(parts,5)['base'],fixture.old)
            self.assertEqual(goal.step_part(parts,50)['base'],fixture.new)

    def test_missing_old5_duplicate_copied_prefix_and_update201_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory)
            old=fixture.old/'logs/actual-updates.jsonl';original=old.read_text()
            old.write_text('\n'.join(original.splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'incomplete'):goal.part_updates(fixture.parts)
            old.write_text(original)
            new=fixture.new/'logs/actual-updates.jsonl';saved=new.read_text()
            new.write_text(json.dumps(dict(step=5,run_id='c'*32))+'\n'+saved)
            with self.assertRaisesRegex(ValueError,'copied prefix'):goal.part_updates(fixture.parts)
            new.write_text(saved+json.dumps(dict(step=201,run_id='c'*32))+'\n')
            with self.assertRaisesRegex(ValueError,'update201'):goal.part_updates(fixture.parts)

    def test_gap_overlap_wrong_manifest_and_changed_science_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory)
            for start in (5,7):
                parts=copy.deepcopy(fixture.parts);parts[1]['first']=start
                with fixture.provider(parts),self.assertRaisesRegex(ValueError,'overlap, gap'):
                    goal.history_parts(fixture.new,fixture.run)
            parts=copy.deepcopy(fixture.parts);parts[0]['run']['source_commit']='3'*40
            with fixture.provider(parts),self.assertRaisesRegex(ValueError,'manifest differs'):
                goal.history_parts(fixture.new,fixture.run)
            parts=copy.deepcopy(fixture.parts);parts[0]['run']['training_world_size']=8
            (fixture.old/'logs/run-manifest.json').write_text(json.dumps(parts[0]['run']))
            with fixture.provider(parts),self.assertRaisesRegex(ValueError,'scientific/data/topology'):
                goal.history_parts(fixture.new,fixture.run)

    def test_all200_original_selection_commit_paths_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory);fixture.attempts()
            with patch.object(goal,'verify_selection',return_value={'measured':True}) as selection,\
                 patch.object(goal,'verify_formal_part',side_effect=lambda part,rows:dict(base=str(part['base']),first=rows[0]['step'],last=rows[-1]['step'])):
                attempts,hashes,formal=goal.verify_attempt_parts(fixture.parts)
            self.assertEqual(len(attempts),200);self.assertEqual(selection.call_count,200)
            self.assertEqual(hashes[4]['producer_base'],str(fixture.old));self.assertEqual(hashes[5]['producer_base'],str(fixture.new))
            self.assertEqual([(v['first'],v['last']) for v in formal],[(1,5),(6,200)])

    def test_teacher_is_read_only_from_actual_original_fresh_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory)
            (fixture.new/'logs/full-run-acceptance.json').write_text(json.dumps(dict(
                structural_accepted=True,actual_outer_updates=200,run_id='c'*32)))
            with fixture.provider(),patch.object(goal.os,'getuid',return_value=14090),\
                 patch.object(goal,'verify_teacher',side_effect=RuntimeError('stop after teacher routing')) as teacher,\
                 patch.object(goal.subprocess,'check_output') as query:
                with self.assertRaisesRegex(RuntimeError,'stop after teacher routing'):
                    goal.verify(fixture.new,fixture.new/'audit')
                teacher.assert_called_once_with(fixture.old,fixture.parent);query.assert_not_called()

    def test_incomplete_suffix_hardfails_before_loading_any_restore_source(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory);ledger=fixture.new/'logs/actual-updates.jsonl'
            rows=[json.loads(x) for x in ledger.read_text().splitlines()][:-1]
            fixture.lines(ledger,rows)
            with patch.object(goal.os,'getuid',return_value=14090),\
                 patch.object(goal,'history_parts') as sources,patch.object(goal,'verify_teacher') as teacher,\
                 patch.object(goal.subprocess,'check_output') as query:
                with self.assertRaisesRegex(ValueError,'completed200 suffix missing'):
                    goal.verify(fixture.new,fixture.new/'audit')
                sources.assert_not_called();teacher.assert_not_called();query.assert_not_called()

    def test_historical_fixed5_and_current_milestones_final_are_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory)
            def specification(count):
                sources=['source'+str(i%7) for i in range(count)]
                record=dict(sha256='v' if count==512 else 'f',
                    source_counts={name:sources.count(name) for name in sorted(set(sources))})
                frame={'data_source':SimpleNamespace(tolist=lambda:sources)}
                return record,frame,['qid'+str(i) for i in range(count)]
            def row(step,final=False,correct=0):
                count=51713 if final else 512;record,frame,ids=specification(count)
                em=[1]*correct+[0]*(count-correct)
                return dict(run_id='c'*32,step=step,final=final,split='final' if final else 'validation',
                    question_ids=ids,data_sources=frame['data_source'].tolist(),validation_sha256=record['sha256'],
                    em=em,f1=[float(x) for x in em],correct=correct,weighted_em=correct/count,
                    weighted_f1=correct/count,counts=record['source_counts'])
            fixture.lines(fixture.old/'logs/validation-identities.jsonl',[row(5,correct=101)])
            fixture.lines(fixture.new/'logs/validation-identities.jsonl',[row(i) for i in (50,100,150,200)])
            (fixture.new/'logs/final-identity.json').write_text(json.dumps(row(200,final=True)))
            with patch('reproduction.common8.validation_identity.expected',side_effect=lambda base,split:specification(51713 if split=='final' else 512)):
                scores,final,em,f1=goal.verify_evaluations(fixture.new,fixture.run,parts=fixture.parts)
                self.assertEqual(scores['5']['correct'],101)
                self.assertEqual(scores['5']['producer_base'],str(fixture.old))
                self.assertEqual(scores['200']['producer_base'],str(fixture.new))
                self.assertEqual(len(final['em']),51713);self.assertEqual(len(em),7);self.assertEqual(len(f1),7)
                # A new row5 cannot replace missing genuine original row5.
                fixture.lines(fixture.old/'logs/validation-identities.jsonl',[])
                fixture.lines(fixture.new/'logs/validation-identities.jsonl',[row(i) for i in (5,50,100,150,200)])
                with self.assertRaisesRegex(ValueError,'fixed512 validation at5'):
                    goal.verify_evaluations(fixture.new,fixture.run,parts=fixture.parts)

    def test_formal_prefix_uses_original_plan_source_and_four_rank_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory);part=fixture.parts[0]
            updates=[json.loads(x) for x in (fixture.old/'logs/actual-updates.jsonl').read_text().splitlines()]
            artifact={'original_plan':True}
            (fixture.old/'logs/policy-safety-calibration.json').write_text(json.dumps(artifact))
            with patch('reproduction.common8.behavior_runtime.read_gate_receipt',return_value={'proof_sha256':'g'}) as gates,\
                 patch('reproduction.common8.behavior_runtime.verify_write_receipt_links') as links,\
                 patch('reproduction.common8.runtime.budget_rows') as budgets,\
                 patch('reproduction.common8.worker_evidence.read_console_form',return_value='original-console'),\
                 patch('reproduction.common8.runtime.verify_scheduler_receipts',return_value={str(i):{'rank':i} for i in range(4)}) as schedulers,\
                 patch('reproduction.common8.formal_inline_safety.verify_plan',return_value=({},'original-plan-sha')) as plan,\
                 patch('reproduction.common8.formal_attempt_audit.verify_attempt_history',return_value={'structural_accepted':True,'completed_outer_count':5}) as monitor:
                proof=goal.verify_formal_part(part,updates)
                self.assertEqual(proof['source_commit'],'1'*40)
                self.assertEqual([call.args[2] for call in gates.call_args_list],list(range(1,6)))
                self.assertTrue(all(call.args[:2]==(fixture.old,fixture.parent) for call in gates.call_args_list))
                self.assertEqual(links.call_args.args[2:],(1,5))
                self.assertEqual(schedulers.call_args.args[2:4],(1,5))
                self.assertEqual(budgets.call_args.args[:3],('original-console',1,5))
                plan.assert_called_once_with(fixture.old,artifact,fixture.parent)
                self.assertEqual(monitor.call_args.kwargs,dict(last=5,plan_sha256='original-plan-sha'))
                self.assertEqual(set(monitor.call_args.args[3]),{'0','1','2','3'})
                monitor.return_value={'structural_accepted':False,'completed_outer_count':5}
                with self.assertRaisesRegex(ValueError,'formal attempt audit failed'):
                    goal.verify_formal_part(part,updates)

    def test_attempt_uuid_reuse_bad_head_commit_and_scheduler201_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=HistoryFixture(directory);fixture.attempts()
            with patch.object(goal,'verify_selection',return_value={'measured':True}),\
                 patch.object(goal,'verify_formal_part',return_value={}):
                event=fixture.new/'artifacts/question-probe/completed'/f'{6:032x}.json'
                original=event.read_text();bad=json.loads(original);bad['head_update']['update_count']=5
                event.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'head updated outside'):goal.verify_attempt_parts(fixture.parts)
                event.write_text(original)
                scheduler=fixture.old/'logs/scheduler-rank-0.jsonl'
                fixture.lines(scheduler,[{'logical_update':201}])
                with self.assertRaisesRegex(ValueError,'update201'):goal.verify_attempt_parts(fixture.parts)
                fixture.lines(scheduler,[{'logical_update':1}])
                ledger=fixture.new/'logs/actual-attempts.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines()]
                rows[0]['attempt_index']=5;fixture.lines(ledger,rows)
                with self.assertRaisesRegex(ValueError,'identity, chronology'):goal.verify_attempt_parts(fixture.parts)


if __name__=='__main__':unittest.main()
