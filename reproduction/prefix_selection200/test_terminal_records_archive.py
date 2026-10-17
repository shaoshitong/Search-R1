"""Tiny CPU closure checks: real hardlinks, source isolation and QA identity."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from .terminal_records_archive import RUN, FileInventory, assert_terminal, assert_final_rows, canonical_rows


class TerminalArchiveTests(unittest.TestCase):
    def boundary(self):
        return [dict(run_id=RUN,outers=200,base='/mnt/data/shaoshitong/run'),
            dict(run_id=RUN,outer=200,actor=dict(actual_adam_steps_per_rank=[4000]*8,scheduler_completed_outer=200),probe=dict(actual_adam_steps_per_rank=[1000]*8)),
            dict(run_id=RUN,complete_outers=200),
            dict(run_id=RUN,cleaned=True,owned_births_all_gone=False,remaining_owned_births=[123]),
            dict(run_id=RUN,source_base='/mnt/data/shaoshitong/run',status='NATIVE_COMPLETED_ZERO_OWNED_RELEASE_PROVED',
                native=dict(JobState='COMPLETED',ExitCode='0:0'),owned_release=dict(two_actual_checks=True,
                all_original_births_gone=True,cgroup_gone=True,own_gpu_processes=[]),
                evidence=[dict(role=r)for r in ['native_terminal','owned_release_first','owned_release_second']])]

    def test_partial_running_or_leaked_resources_cannot_seal(self):
        args=self.boundary();assert_terminal(*args)  # A later actual release resolves an earlier transient zombie.
        for index,change in [(1,lambda x:x.update(outer=199)),(4,lambda x:x['native'].update(JobState='RUNNING')),
                             (4,lambda x:x['owned_release'].update(own_gpu_processes=[17])),
                             (4,lambda x:x.update(evidence=[]))]:
            args=self.boundary();change(args[index])
            with self.assertRaises(ValueError):assert_terminal(*args)

    def test_exact_question_source_and_ordinal_not_only_total_count(self):
        canonical=canonical_rows([dict(data_source='a',qid='one'),dict(data_source='b',qid='two'),dict(data_source='b',qid='three')])
        actual=[dict(r,em=int(i!=2),f1=float(i!=2))for i,r in enumerate(canonical)]
        summary=dict(qa_count=3,em=2/3,f1=2/3,per_source=dict(a=dict(qa_count=1,em=1.,f1=1.),b=dict(qa_count=2,em=.5,f1=.5)))
        self.assertEqual(assert_final_rows(canonical,list(reversed(actual)),summary,expected_count=3,expected_sources=2),{'a':1,'b':2})
        for field,value in [('qid','one'),('data_source','a'),('evaluation_ordinal',0)]:
            bad=copy.deepcopy(actual);bad[2][field]=value
            with self.assertRaises(ValueError):assert_final_rows(canonical,bad,summary,expected_count=3,expected_sources=2)
        bad=copy.deepcopy(summary);bad['em']=1.
        with self.assertRaises(ValueError):assert_final_rows(canonical,actual,bad,expected_count=3,expected_sources=2)
        summary['per_source']['a']['qa_count']=2;summary['per_source']['b']['qa_count']=1
        with self.assertRaises(ValueError):assert_final_rows(canonical,actual,summary,expected_count=3,expected_sources=2)

    def test_actual_hardlink_dedup_preserves_both_original_names(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder).resolve();one=root/'first';two=root/'second';one.write_bytes(b'original trajectory ids and labels')
            os.link(one,two);manifest=FileInventory([root]);manifest.add(one,'first');manifest.add(two,'inherited')
            rows=manifest.files();self.assertEqual(len(rows),1);self.assertEqual(set(rows[0]['original_paths']),{str(one),str(two)})
            link=root/'symlink';link.symlink_to(one)
            with self.assertRaises(ValueError):manifest.add(link,'unverified-alias')
            with self.assertRaises(ValueError):manifest.add(root/'..'/'escape','outside')

    def test_closed_failed_partial_records_are_preserved(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder).resolve();record=root/'trajectory.jsonl.partial';record.write_text('original failed attempt')
            derived=root/'rank-0-actor-hidden.npy.partial';derived.write_bytes(b'consumed derived hidden buffer')
            m=FileInventory([root]);m.tree(root,'history');self.assertEqual([x['source_path']for x in m.files()],[str(record)])


if __name__=='__main__':unittest.main()
