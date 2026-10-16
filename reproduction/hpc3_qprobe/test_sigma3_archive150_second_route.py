"""Explicit second150/V2 final route retains S,5f and every final gate."""
import base64
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader
from reproduction.hpc3_qprobe import independent_goal_control_sigma3 as holder
from reproduction.hpc3_qprobe import recovery_lineage_stage as lineage
from reproduction.hpc3_qprobe import test_sigma3_archive150_route as first_tests
from reproduction.hpc3_qprobe.test_independent_goal_acceptance_sigma3 import write

PINS=first_tests.PINS[:-1]+(reader.ARCHIVE150_SECOND_RECOVERY_SOURCE,)


def actual_chain():
    fixture=json.loads(Path(__file__).with_name('archive150_second_failure_fixture.json').read_bytes())
    folder=str(reader.ROOT/'followthrough/721485-archive150-recovery-first-20261016')
    first=json.loads(base64.b64decode(fixture[folder]['claim.json']['raw_base64']))['recovery']
    old=first['prior_archive100_fourth_recovery']
    latest=dict(first,source={'commit':PINS[-1]},prior_archive150_first_recovery=first,
        original_archive_child_exit=-15,storage_protocol='actual-full-owner-partition-du-v2',
        active_CPU_retirement=dict(source={'commit':'5e519ad272660c97578447f06b0190e30b22517b'},
            complete={'sha256':'6a28843d779c5feb69d286ad19f22a5a75a12b93ac469b71bd880841a056be38'},
            active_retirement=True,spontaneous_failure=False,actual_archive_child_wait_exit_code=-15,original_follower_exit_code=1))
    return old,first,latest


class ExplicitSecond150(unittest.TestCase):
    def test_route_and_scope_are_separate_and_exact(self):
        name,folder=reader.followthrough_route(PINS[-1],*PINS)
        self.assertEqual((name,folder.name),('recover_sigma3_archive150_second','721485-archive150-recovery-second-20261016'))
        self.assertNotEqual(holder.scope(*PINS),holder.scope(*first_tests.PINS))
        self.assertTrue(holder.scope(*PINS).name.endswith('-archive150-recovery-second'))
        self.assertEqual(reader.ARCHIVE150_ROUTES[PINS[-1]][2:],(
            'owner_partition_storage_v2','actual-full-owner-partition-du-v2'))
        for i in range(len(PINS)):
            bad=list(PINS);bad[i]='a'*40
            with self.subTest(i=i),self.assertRaises(ValueError):reader.followthrough_route(PINS[-1],*bad)

    def test_original_s_and_5f_are_consumed_separately(self):
        old,first,latest=actual_chain();expected={PINS[3]:old,reader.ARCHIVE150_FIRST_RECOVERY_SOURCE:first}
        with patch.object(lineage,'consume',side_effect=lambda commit,name:expected[commit]) as consume:
            self.assertEqual(reader.prior_archive150_evidence(latest,PINS[-1],PINS[3]),old)
            self.assertEqual([call.args for call in consume.call_args_list],[(PINS[3],'recover_sigma3_archive100_fourth'),
                (reader.ARCHIVE150_FIRST_RECOVERY_SOURCE,'recover_sigma3_archive150_first')])
        for key,value in [('prior_archive150_first_recovery',{}),('original_archive_child_exit',1),
                          ('storage_protocol','actual-full-owner-partition-du-v1'),('active_CPU_retirement',{})]:
            changed=copy.deepcopy(latest);changed[key]=value
            with self.subTest(key=key),patch.object(lineage,'consume',side_effect=lambda commit,name:expected[commit]),self.assertRaises(ValueError):
                reader.prior_archive150_evidence(changed,PINS[-1],PINS[3])

    def test_missing200_fails_before_source_walk_creation_or_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();base=root/'run';(base/'logs').mkdir(parents=True)
            (base/'logs/actual-updates.jsonl').write_text('{"step":170}\n');before=list(root.rglob('*'))
            with patch.object(holder,'DIRECTORY',root/'new'),patch.object(holder,'caps'),\
                 patch.object(reader,'BASE',base),patch.object(reader.shared,'OWNER',os.getuid()),\
                 patch.object(holder.ctl,'source') as source,patch.object(holder.subprocess,'Popen') as spawn:
                with self.assertRaisesRegex(ValueError,'fresh1..200'):
                    holder.start('a'*40,followthrough_commit=PINS[-1],recovery_commit=PINS[0],observer_recovery_commit=PINS[1],
                        archive_recovery_commit=PINS[2],archive100_recovery_commit=PINS[3],archive150_recovery_commit=PINS[4])
                source.assert_not_called();spawn.assert_not_called()
            self.assertEqual(before,list(root.rglob('*')))

    def test_native_source_is_second_follower_with_equal_science(self):
        run={'source_sha256':{'search_g1/science.py':'a'*64}};native={'audit_commit':PINS[-1],'audit_id':'native'}
        audit={'source_commit':PINS[-1],'run_id':reader.RUN,'final_restore_audit':{'audit_id':'native'},'source_sha256':run['source_sha256']}
        reader.native_source_gate(run,native,audit,{'files':run['source_sha256']},PINS[-1],*PINS)
        with self.assertRaisesRegex(ValueError,'native audit source'):
            reader.native_source_gate(run,dict(native,audit_commit=PINS[3]),audit,{'files':run['source_sha256']},PINS[-1],*PINS)

    def test_final_stdout_requires_v2_and_preserved_s_retirement(self):
        commit,admission,proof=first_tests.Explicit150().output();old,first,latest=actual_chain()
        for value in (admission,proof):value['archive150_recovery_commit']=PINS[-1]
        admission.update(followthrough_source=latest['source'],archive150_recovery=latest)
        proof['followthrough_commit']=PINS[-1];follow=proof['actual_followthrough_full_CPU_consumer']
        follow.update(source=latest['source'],archive150_recovery=copy.deepcopy(latest))
        for step in ('150','200'):proof['actual_archive_and_submit_held_children'][step]['owner_partition_storage']['protocol']='actual-full-owner-partition-du-v2'
        holder.accept_output(proof,commit,PINS[-1],admission,*PINS)
        for step in ('150','200'):
            bad=copy.deepcopy(proof);bad['actual_archive_and_submit_held_children'][step]['owner_partition_storage']['protocol']='actual-full-owner-partition-du-v1'
            with self.subTest(step=step),self.assertRaisesRegex(ValueError,'complete owner partition'):
                holder.accept_output(bad,commit,PINS[-1],admission,*PINS)
        for key,value in [('actual_native_two_readbacks_reconsumed',False),('update201_performed',True)]:
            bad=copy.deepcopy(proof);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):holder.accept_output(bad,commit,PINS[-1],admission,*PINS)
        changed=copy.deepcopy(admission);changed['archive150_recovery']['prior_archive150_first_recovery']={}
        with self.assertRaisesRegex(ValueError,'original S recovery'):holder.accept_output(proof,commit,PINS[-1],changed,*PINS)


class ActualHeldSecond150(first_tests.ActualHeld150):
    def setUp(self):
        with patch.object(first_tests,'PINS',PINS):super().setUp()
        for step in (150,200):
            folder=self.folders[step];path=folder/'submit-claim.json';claim=json.loads(path.read_bytes())
            claim['storage_wait_protocol']='actual-full-owner-partition-du-v2';proof=write(path,claim)
            for name in ('runner-binding.json','task-binding.json','task-result.json'):
                path=folder/name;value=json.loads(path.read_bytes());value['claim_sha256']=proof['sha256'];last=write(path,value)
            path=folder/'worker-result.json';value=json.loads(path.read_bytes())
            value.update(claim_sha256=proof['sha256'],task_result_sha256=last['sha256']);write(path,value)
            self.storage[step]['protocol']='actual-full-owner-partition-du-v2'
        partition=SimpleNamespace(PROTOCOL='actual-full-owner-partition-du-v2',consume=lambda folder:copy.deepcopy(self.storage[int(folder.name.split('-')[-1])]))
        legacy=SimpleNamespace(service=lambda:SimpleNamespace(inherited=lambda:{'existing_archive_claims':{'5':self.inherited['5']}}))
        self.loader=self.stack.enter_context(patch.object(reader,'load_fixed',side_effect=lambda commit,name,*_:(partition if name=='owner_partition_storage_v2' else legacy,{})))

    def consume(self):
        return reader.held_archive_evidence({},self.producer,self.follow,self.sources[150],self.controllers[150],
            observer_recovery_commit=PINS[1],archive_recovery_commit=PINS[2],archive100_recovery_commit=PINS[3],archive150_recovery_commit=PINS[4])

    def test_exact_v2_module_is_loaded(self):
        result,_=self.consume()
        self.assertEqual(result['200']['owner_partition_storage']['protocol'],'actual-full-owner-partition-du-v2')
        self.assertIn((PINS[-1],'owner_partition_storage_v2','ctl'),[call.args for call in self.loader.call_args_list])


if __name__=='__main__':unittest.main()
