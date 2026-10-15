"""Explicit100 lineage and real small held receipts; no remote completion claim."""
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
from reproduction.hpc3_qprobe.test_independent_goal_acceptance_sigma3 import write
from reproduction.hpc3_qprobe.test_independent_goal_control_sigma3 import output_fixture

PINS=(reader.ARCHIVE5_RECOVERY_SOURCE,'f979d53cdb0d348e87c5877d84403252e85cd784',
      reader.ARCHIVE50_SECOND_RECOVERY_SOURCE,reader.ARCHIVE100_RECOVERY_SOURCE)


class Exact100Route(unittest.TestCase):
    def test_second100_has_exact_source_module_and_distinct_follower_and_holder_scope(self):
        pins=PINS[:-1]+(reader.ARCHIVE100_SECOND_RECOVERY_SOURCE,)
        name,folder=reader.followthrough_route(pins[-1],*pins)
        self.assertEqual(name,'recover_sigma3_archive100_second')
        self.assertEqual(folder.name,'721485-archive100-recovery-second-20261016')
        self.assertTrue(holder.scope(*pins).name.endswith('-archive100-recovery-second'))
        self.assertNotEqual(holder.scope(*pins),holder.scope(*PINS))
        self.assertEqual(holder.recovery_args(*pins)[-2:],['--archive100-recovery-commit',pins[-1]])
        with self.assertRaises(ValueError):reader.followthrough_route(PINS[-1],*pins)
        with self.assertRaises(ValueError):reader.followthrough_route(pins[-1],*PINS)
        for index in range(4):
            bad=list(pins);bad[index]='a'*40
            with self.subTest(index=index),self.assertRaises(ValueError):reader.followthrough_route(pins[-1],*bad)

    def test_second100_reconsumes_c1_once_and_rejects_prior_or_inheritance_retagging(self):
        prior50={'source':{'commit':reader.ARCHIVE50_FIRST_RECOVERY_SOURCE}}
        fifty={'source':{'commit':PINS[2]},'prior_archive50_recovery':prior50,
               'observer_recovery':{'raw':'f979'},'legacy_archive5_recovery':{'five':'09cc'}}
        first={'source':{'commit':PINS[-1]},'prior_archive50_second_recovery':fifty,
               'observer_recovery':fifty['observer_recovery'],'legacy_archive5_recovery':fifty['legacy_archive5_recovery'],
               'inherited_archive50':{'original':'B89450'}}
        second=dict(first,source={'commit':reader.ARCHIVE100_SECOND_RECOVERY_SOURCE},prior_archive100_recovery=first)
        values={PINS[-1]:first,PINS[2]:fifty,reader.ARCHIVE50_FIRST_RECOVERY_SOURCE:prior50}
        def load(commit,*_):return SimpleNamespace(consume=lambda _:values[commit]),{}
        with patch.object(reader,'load_fixed',side_effect=load) as loads:
            measured=reader.archive50_recovery_evidence(second,PINS[2],reader.ARCHIVE100_SECOND_RECOVERY_SOURCE,with_prior100=True)
            self.assertEqual(measured,(fifty,first))
            self.assertEqual([c.args[0] for c in loads.call_args_list],list(values))
        for key in ('prior_archive100_recovery','prior_archive50_second_recovery','inherited_archive50','observer_recovery','legacy_archive5_recovery'):
            bad=copy.deepcopy(second);bad[key]={}
            with self.subTest(key=key),patch.object(reader,'load_fixed',side_effect=load),self.assertRaises(ValueError):
                reader.archive50_recovery_evidence(bad,PINS[2],reader.ARCHIVE100_SECOND_RECOVERY_SOURCE)
        for broken in values:
            def changed(commit,*_):return SimpleNamespace(consume=lambda _:{} if commit==broken else values[commit]),{}
            with self.subTest(actual_changed=broken),patch.object(reader,'load_fixed',side_effect=changed),self.assertRaises(ValueError):
                reader.archive50_recovery_evidence(second,PINS[2],reader.ARCHIVE100_SECOND_RECOVERY_SOURCE)
        with self.assertRaises(ValueError):reader.prior_archive100_evidence(second,PINS[-1])

    def test_second100_final_output_requires_separate_c1_and_original50raw_boundaries(self):
        pins=PINS[:-1]+(reader.ARCHIVE100_SECOND_RECOVERY_SOURCE,)
        commit='a'*40;_,admission,proof=output_fixture(commit)
        for value in (admission,proof):
            value.update(recovery_commit=pins[0],observer_recovery_commit=pins[1],
                archive_recovery_commit=pins[2],archive100_recovery_commit=pins[3])
        first={'source':{'commit':PINS[-1]},'preserved_failure':'C1 actual held1'}
        second={'source':{'commit':pins[-1]},'prior_archive100_recovery':first}
        admission.update(followthrough_source={'commit':pins[-1]},raw_source={'commit':pins[1]},
            archive5_recovery={'five':'09cc'},observer_recovery={'raw':'f979'},archive50_recovery={'fifty':'B894'},
            archive50_prior_recovery={'prior':'dc9'},archive100_recovery=second,archive100_prior_recovery=first)
        proof['followthrough_commit']=pins[-1];follow=proof['actual_followthrough_full_CPU_consumer']
        follow['source']=admission['followthrough_source']
        for key in ('archive5_recovery','observer_recovery','archive50_recovery','archive50_prior_recovery','archive100_recovery','archive100_prior_recovery'):
            follow[key]=admission[key]
        proof['independent_all_normal_return_raw']['source']=admission['raw_source']
        holder.accept_output(proof,commit,pins[-1],admission,*pins)
        for key in ('archive100_prior_recovery','archive50_recovery','observer_recovery'):
            bad=copy.deepcopy(proof);bad['actual_followthrough_full_CPU_consumer'][key]={}
            with self.subTest(key=key),self.assertRaises(ValueError):holder.accept_output(bad,commit,pins[-1],admission,*pins)
        bad=copy.deepcopy(admission);bad.pop('archive100_prior_recovery')
        with self.assertRaisesRegex(ValueError,'authenticated C1'):holder.accept_output(proof,commit,pins[-1],bad,*pins)
        bad=copy.deepcopy(admission);bad['archive100_recovery']['prior_archive100_recovery']={}
        with self.assertRaisesRegex(ValueError,'authenticated C1'):holder.accept_output(proof,commit,pins[-1],bad,*pins)

    def test_second100_missing200_still_refuses_before_any_source_scope_or_spawn(self):
        with patch(__name__+'.PINS',PINS[:-1]+(reader.ARCHIVE100_SECOND_RECOVERY_SOURCE,)):
            self.test_less_than200_refuses_before_scope_source_or_spawn_on_new_route()

    def test_distinct100_route_scope_and_cli_preserve_all_four_prior_pins(self):
        name,folder=reader.followthrough_route(PINS[-1],*PINS)
        self.assertEqual(name,'recover_sigma3_archive100')
        self.assertEqual(folder.name,'721485-archive100-recovery-20261016')
        self.assertTrue(holder.scope(*PINS).name.endswith('-archive100-recovery'))
        self.assertNotEqual(holder.scope(*PINS),holder.scope(*PINS[:-1]))
        self.assertEqual(holder.recovery_args(*PINS)[-2:],['--archive100-recovery-commit',PINS[-1]])
        for index in range(4):
            bad=list(PINS);bad[index]='a'*40
            with self.subTest(index=index),self.assertRaises(ValueError):reader.followthrough_route(PINS[-1],*bad)
        with self.assertRaises(ValueError):reader.followthrough_route(PINS[2],*PINS)

    def test_original_b894_and_dc9_each_reconsumed_not_only_embedded_json(self):
        prior={'source':{'commit':reader.ARCHIVE50_FIRST_RECOVERY_SOURCE}}
        fifty={'source':{'commit':PINS[2]},'prior_archive50_recovery':prior,
               'observer_recovery':{'raw':'f979'},'legacy_archive5_recovery':{'five':'09cc'}}
        hundred={'source':{'commit':PINS[-1]},'prior_archive50_second_recovery':fifty,
                 'observer_recovery':fifty['observer_recovery'],'legacy_archive5_recovery':fifty['legacy_archive5_recovery']}
        def load(commit,*_):return SimpleNamespace(consume=lambda _:fifty if commit==PINS[2] else prior),{}
        with patch.object(reader,'load_fixed',side_effect=load) as loads:
            self.assertEqual(reader.archive50_recovery_evidence(hundred,PINS[2],PINS[-1]),fifty)
            self.assertEqual([c.args[0] for c in loads.call_args_list],[PINS[2],reader.ARCHIVE50_FIRST_RECOVERY_SOURCE])
        for broken in (PINS[2],reader.ARCHIVE50_FIRST_RECOVERY_SOURCE):
            def changed(commit,*_):return SimpleNamespace(consume=lambda _:{} if commit==broken else fifty if commit==PINS[2] else prior),{}
            with self.subTest(commit=broken),patch.object(reader,'load_fixed',side_effect=changed),self.assertRaises(ValueError):
                reader.archive50_recovery_evidence(hundred,PINS[2],PINS[-1])
        bad=copy.deepcopy(hundred);bad['observer_recovery']={}
        with patch.object(reader,'load_fixed',side_effect=load),self.assertRaises(ValueError):
            reader.archive50_recovery_evidence(bad,PINS[2],PINS[-1])

    def test_final_stdout_binds_separate10050raw_and_actual_second_native_readback(self):
        commit='a'*40;_,admission,proof=output_fixture(commit)
        for value in (admission,proof):
            value.update(recovery_commit=PINS[0],observer_recovery_commit=PINS[1],
                archive_recovery_commit=PINS[2],archive100_recovery_commit=PINS[3])
        proof['followthrough_commit']=PINS[-1]
        admission.update(followthrough_source={'commit':PINS[-1]},raw_source={'commit':PINS[1]},
            archive5_recovery={'five':'09cc'},observer_recovery={'raw':'f979'},
            archive50_recovery={'fifty':'B894'},archive50_prior_recovery={'prior':'dc9'},
            archive100_recovery={'source':{'commit':PINS[-1]},'hundred':'separate'})
        follow=proof['actual_followthrough_full_CPU_consumer']
        follow['source']=admission['followthrough_source']
        for key in ('archive5_recovery','observer_recovery','archive50_recovery','archive50_prior_recovery','archive100_recovery'):
            follow[key]=admission[key]
        proof['independent_all_normal_return_raw']['source']=admission['raw_source']
        holder.accept_output(proof,commit,PINS[-1],admission,*PINS)
        for key,value in (('archive100_recovery_commit',PINS[2]),('actual_native_two_readbacks_reconsumed',False),('update201_performed',True)):
            bad=copy.deepcopy(proof);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):holder.accept_output(bad,commit,PINS[-1],admission,*PINS)
        bad=copy.deepcopy(proof);bad['actual_followthrough_full_CPU_consumer']['archive50_recovery']={'fifty':'new100'}
        with self.assertRaises(ValueError):holder.accept_output(bad,commit,PINS[-1],admission,*PINS)
        bad=copy.deepcopy(proof);bad['actual_followthrough_full_CPU_consumer']['archive100_recovery']={}
        with self.assertRaises(ValueError):holder.accept_output(bad,commit,PINS[-1],admission,*PINS)
        bad=copy.deepcopy(proof);bad['independent_all_normal_return_raw']['source']={'commit':PINS[-1]}
        with self.assertRaises(ValueError):holder.accept_output(bad,commit,PINS[-1],admission,*PINS)

    def test_less_than200_refuses_before_scope_source_or_spawn_on_new_route(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();base=root/'run';(base/'logs').mkdir(parents=True)
            (base/'logs/actual-updates.jsonl').write_text('{"step":102}\n')
            scope=root/'independent';before=sorted(str(p) for p in root.rglob('*'))
            with patch.object(holder,'DIRECTORY',scope),patch.object(holder,'caps'),\
                 patch.object(reader,'BASE',base),patch.object(reader.shared,'OWNER',os.getuid()),\
                 patch.object(holder.ctl,'source') as source,patch.object(holder.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'fresh1..200'):
                    holder.start('a'*40,followthrough_commit=PINS[-1],recovery_commit=PINS[0],
                        observer_recovery_commit=PINS[1],archive_recovery_commit=PINS[2],archive100_recovery_commit=PINS[3])
                source.assert_not_called();launch.assert_not_called()
            self.assertEqual(before,sorted(str(p) for p in root.rglob('*')))


class InheritedHeldReceipts(unittest.TestCase):
    def test_second100_still_requires_original5_and50_allfive_held0_and_native_source(self):
        with patch(__name__+'.PINS',PINS[:-1]+(reader.ARCHIVE100_SECOND_RECOVERY_SOURCE,)):
            self.test_exact_original5_and50_and_new100150200_native_sources_are_required()

    def test_exact_original5_and50_and_new100150200_native_sources_are_required(self):
        with tempfile.TemporaryDirectory() as d,patch.object(reader,'OWNER',os.getuid()),\
             patch.object(reader.shared,'OWNER',os.getuid()):
            root=Path(d).resolve();producer={'run_id':reader.RUN};pair={'sha256':'b'*64}
            sources={5:{'commit':PINS[0]},50:{'commit':PINS[2]},100:{'commit':PINS[-1]}}
            controllers={5:dict(pid=10,start_ticks=100,sid=10,uid=os.getuid()),
                         50:dict(pid=20,start_ticks=200,sid=20,uid=os.getuid()),
                         100:dict(pid=30,start_ticks=300,sid=30,uid=os.getuid())}
            inherited={};folders={}
            with patch.object(reader,'ROOT',root),patch.object(reader.ctl,'ROOT',root),patch.object(reader.ctl,'live',return_value=False):
                for index,step in enumerate((5,50,100,150,200,None)):
                    final=step is None;key=step if step in (5,50) else 100
                    publication=sources[key];controller=controllers[key]
                    folder=root/'final-control'/reader.RUN/('audit-controller' if final else 'archive-'+str(step));folders[step]=folder
                    token='%032x'%(index+1);role='detached_cpu_final_audit_submission' if final else 'detached_cpu_pair_archive'
                    claim=dict(schema_version=1,role=role,token=token,producer=producer,source=publication,
                        controller=controller,created_unix=1,training_permitted=False,direct_gpu_device_access_permitted=False,
                        native_gpu_audit_submission_permitted=final,pair=pair)
                    if not final:claim['step']=step
                    if PINS[-1]==reader.ARCHIVE100_FOURTH_RECOVERY_SOURCE and step in (100,150,200):
                        claim['storage_wait_protocol']='actual-full-du-repair-v1'
                    claim['command']=reader.ctl.command(root/'source'/publication['commit'],
                        'audit-worker' if final else 'archive-worker',folder/'submit-claim.json',token)
                    claim_file=write(folder/'submit-claim.json',claim)
                    runner=dict(pid=200+index,start_ticks=2000+index,sid=200+index,uid=os.getuid())
                    child=dict(pid=300+index,start_ticks=3000+index,sid=runner['sid'],uid=os.getuid())
                    binding=dict(token=token,claim_sha256=claim_file['sha256'])
                    write(folder/'runner-binding.json',dict(binding,actual_process=runner,controller=controller))
                    write(folder/'task-binding.json',dict(binding,actual_process=child,supervisor=runner))
                    task=dict(binding,status='CPU_FINAL_SUBMIT_CONTROLLER_RETURNED' if final else 'CPU_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',
                        actual_process=child,training_performed=False,direct_gpu_device_access_performed=False,
                        actor_model_loaded=False,native_gpu_audit_submission_attempted=final,completed_unix=2)
                    if final:task['submission']={'status':'ACTUAL_SLURM_SUBMITTED'}
                    else:
                        path=root/'archives'/reader.RUN/('global_step_'+str(step))/'complete.json'
                        transport=write(path,dict(run_id=reader.RUN,step=step,world_size=4))
                        task['archive']=dict(path=str(path),sha256=transport['sha256'])
                    task_file=write(folder/'task-result.json',task)
                    result=dict(binding,actual_task_birth=child,actual_supervisor=runner,role=role,actual_task_exit_code=0,
                        status='ACTUAL_CPU_FINAL_SUBMIT_CHILD_COMPLETED0' if final else 'ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0',
                        task_result_sha256=task_file['sha256'],finished_unix=3)
                    result_file=write(folder/'worker-result.json',result)
                    if step in (5,50):
                        inherited[str(step)]=dict(path=claim_file['path'],sha256=claim_file['sha256'],source=publication,controller=controller,
                            worker_result_sha256=result_file['sha256'],task_result_sha256=task_file['sha256'],transport_sha256=transport['sha256'])
                recovery={'inherited_archive50':inherited['50']}
                follow=SimpleNamespace(consume=lambda _:recovery,
                    service=lambda:SimpleNamespace(inherited=lambda:{'existing_archive_claims':{'5':inherited['5']}}),
                    control=SimpleNamespace(pair=lambda *_:(None,pair),inspect_audit=lambda _:dict(
                        status='ACTUAL_SLURM_SUBMITTED',job='800001',audit_base=str(root/'native'))))
                def publication(commit):return None,next(v for v in sources.values() if v['commit']==commit)
                with patch.object(reader,'publication',side_effect=publication),patch.object(reader,'archive50_recovery_evidence'):
                    def consume():return reader.held_archive_evidence({},producer,follow,sources[100],controllers[100],
                        observer_recovery_commit=PINS[1],archive_recovery_commit=PINS[2],archive100_recovery_commit=PINS[3])
                    self.assertEqual(set(consume()[0]),{'5','50','100','150','200','native_submit'})
                    if PINS[-1]==reader.ARCHIVE100_FOURTH_RECOVERY_SOURCE:
                        for storage_step in (100,150,200):
                            path=folders[storage_step]/'submit-claim.json';original=path.read_bytes()
                            value=json.loads(original);value.pop('storage_wait_protocol');write(path,value)
                            with self.subTest(storage_step=storage_step),self.assertRaisesRegex(ValueError,'explicit actual storage wait protocol'):consume()
                            path.write_bytes(original)
                    for field,bad in (('controller',controllers[100]),('source',sources[100])):
                        path=folders[50]/'submit-claim.json';original=path.read_bytes();value=json.loads(original);value[field]=bad;write(path,value)
                        with self.subTest(field=field),self.assertRaisesRegex(ValueError,'original archive claim changed'):consume()
                        path.write_bytes(original)
                    path=folders[150]/'worker-result.json';original=path.read_bytes();path.unlink()
                    with self.assertRaises(FileNotFoundError):consume()
                    path.write_bytes(original)
                    path=folders[200]/'worker-result.json';value=json.loads(path.read_bytes());value['actual_task_exit_code']=1;write(path,value)
                    with self.assertRaisesRegex(ValueError,'actual held original child0'):consume()


class ExactThird100Route(unittest.TestCase):
    def test_third_pin_module_follower_and_scope_cannot_be_swapped_with_either_prior(self):
        pins=PINS[:-1]+(reader.ARCHIVE100_THIRD_RECOVERY_SOURCE,)
        name,folder=reader.followthrough_route(pins[-1],*pins)
        self.assertEqual((name,folder.name),('recover_sigma3_archive100_third','721485-archive100-recovery-third-20261016'))
        scopes={holder.scope(*(PINS[:-1]+(pin,))) for pin in reader.ARCHIVE100_ROUTES}
        self.assertEqual(len(scopes),4)
        self.assertTrue(holder.scope(*pins).name.endswith('-archive100-recovery-third'))
        for old in (PINS[-1],reader.ARCHIVE100_SECOND_RECOVERY_SOURCE):
            with self.assertRaises(ValueError):reader.followthrough_route(old,*pins)
            with self.assertRaises(ValueError):reader.followthrough_route(pins[-1],*(PINS[:-1]+(old,)))

    def test_stage_reconsumes_exact_c5_with_c1_b894_and_inherited50_unchanged(self):
        prior50={'source':{'commit':reader.ARCHIVE50_FIRST_RECOVERY_SOURCE}}
        fifty={'source':{'commit':PINS[2]},'prior_archive50_recovery':prior50,
               'observer_recovery':{'raw':'f979'},'legacy_archive5_recovery':{'five':'09cc'}}
        first={'source':{'commit':PINS[-1]}}
        second=dict(source={'commit':reader.ARCHIVE100_SECOND_RECOVERY_SOURCE},prior_archive100_recovery=first,
            prior_archive50_second_recovery=fifty,observer_recovery=fifty['observer_recovery'],
            legacy_archive5_recovery=fifty['legacy_archive5_recovery'],inherited_archive50={'original':'B89450'})
        third=dict(second,source={'commit':reader.ARCHIVE100_THIRD_RECOVERY_SOURCE},prior_archive100_second_recovery=second)
        def load(commit,*_):return SimpleNamespace(consume=lambda _:fifty if commit==PINS[2] else prior50),{}
        with patch.object(lineage,'consume',return_value=second) as consume,patch.object(reader,'load_fixed',side_effect=load):
            self.assertEqual(reader.archive50_recovery_evidence(third,PINS[2],reader.ARCHIVE100_THIRD_RECOVERY_SOURCE,with_prior100=True),(fifty,first))
            consume.assert_called_once_with(reader.ARCHIVE100_SECOND_RECOVERY_SOURCE,'recover_sigma3_archive100_second')
        for key in ('prior_archive100_second_recovery','prior_archive100_recovery','prior_archive50_second_recovery','inherited_archive50','observer_recovery','legacy_archive5_recovery'):
            bad=copy.deepcopy(third);bad[key]={}
            with self.subTest(key=key),patch.object(lineage,'consume',return_value=second),patch.object(reader,'load_fixed',side_effect=load),self.assertRaises(ValueError):
                reader.archive50_recovery_evidence(bad,PINS[2],reader.ARCHIVE100_THIRD_RECOVERY_SOURCE)
        with patch.object(lineage,'consume',return_value={}),self.assertRaises(ValueError):
            reader.prior_archive100_evidence(third,reader.ARCHIVE100_THIRD_RECOVERY_SOURCE)

    def test_stdout_binds_both_prior100s_and_original50raw_and_rejects_missing_c5(self):
        pins=PINS[:-1]+(reader.ARCHIVE100_THIRD_RECOVERY_SOURCE,);commit='a'*40
        _,admission,proof=output_fixture(commit)
        for value in (admission,proof):
            value.update(recovery_commit=pins[0],observer_recovery_commit=pins[1],archive_recovery_commit=pins[2],archive100_recovery_commit=pins[3])
        first={'source':{'commit':PINS[-1]}}
        second={'source':{'commit':reader.ARCHIVE100_SECOND_RECOVERY_SOURCE},'prior_archive100_recovery':first}
        third={'source':{'commit':pins[-1]},'prior_archive100_recovery':first,'prior_archive100_second_recovery':second}
        admission.update(followthrough_source={'commit':pins[-1]},raw_source={'commit':pins[1]},
            archive5_recovery={'five':'09cc'},observer_recovery={'raw':'f979'},archive50_recovery={'fifty':'B894'},
            archive50_prior_recovery={'prior':'dc9'},archive100_recovery=third,archive100_prior_recovery=first,
            archive100_second_prior_recovery=second)
        proof['followthrough_commit']=pins[-1];follow=proof['actual_followthrough_full_CPU_consumer']
        follow['source']=admission['followthrough_source']
        for key in ('archive5_recovery','observer_recovery','archive50_recovery','archive50_prior_recovery','archive100_recovery','archive100_prior_recovery','archive100_second_prior_recovery'):
            follow[key]=admission[key]
        proof['independent_all_normal_return_raw']['source']=admission['raw_source']
        holder.accept_output(proof,commit,pins[-1],admission,*pins)
        for key in ('archive100_prior_recovery','archive100_second_prior_recovery','archive50_recovery'):
            bad=copy.deepcopy(proof);bad['actual_followthrough_full_CPU_consumer'][key]={}
            with self.subTest(key=key),self.assertRaises(ValueError):holder.accept_output(bad,commit,pins[-1],admission,*pins)
        bad=copy.deepcopy(admission);bad.pop('archive100_second_prior_recovery')
        with self.assertRaisesRegex(ValueError,'authenticated C5'):holder.accept_output(proof,commit,pins[-1],bad,*pins)
        bad=copy.deepcopy(admission);bad['archive100_second_prior_recovery']['prior_archive100_recovery']={}
        with self.assertRaises(ValueError):holder.accept_output(proof,commit,pins[-1],bad,*pins)

    def test_third200earlyrefusal_and_allfive_held0_sources_still_required(self):
        with patch(__name__+'.PINS',PINS[:-1]+(reader.ARCHIVE100_THIRD_RECOVERY_SOURCE,)):
            Exact100Route.test_less_than200_refuses_before_scope_source_or_spawn_on_new_route(self)
            InheritedHeldReceipts.test_exact_original5_and50_and_new100150200_native_sources_are_required(self)


class ExactFourth100Route(unittest.TestCase):
    def test_exact_pin_module_and_scope_are_distinct_and_cross_routes_refuse(self):
        pins=PINS[:-1]+(reader.ARCHIVE100_FOURTH_RECOVERY_SOURCE,)
        name,folder=reader.followthrough_route(pins[-1],*pins)
        self.assertEqual((name,folder.name),('recover_sigma3_archive100_fourth','721485-archive100-recovery-fourth-20261016'))
        self.assertTrue(holder.scope(*pins).name.endswith('-archive100-recovery-fourth'))
        for old in (PINS[-1],reader.ARCHIVE100_SECOND_RECOVERY_SOURCE,reader.ARCHIVE100_THIRD_RECOVERY_SOURCE):
            with self.assertRaises(ValueError):reader.followthrough_route(old,*pins)
            with self.assertRaises(ValueError):reader.followthrough_route(pins[-1],*(PINS[:-1]+(old,)))

    def chain(self):
        first={'source':{'commit':PINS[-1]}}
        second={'source':{'commit':reader.ARCHIVE100_SECOND_RECOVERY_SOURCE},'prior_archive100_recovery':first}
        third=dict(source={'commit':reader.ARCHIVE100_THIRD_RECOVERY_SOURCE},prior_archive100_recovery=first,
            prior_archive100_second_recovery=second,prior_archive50_second_recovery={'fifty':'B894'},
            inherited_archive50={'original':'50'},observer_recovery={'raw':'f979'},legacy_archive5_recovery={'five':'09cc'})
        fourth=dict(third,source={'commit':reader.ARCHIVE100_FOURTH_RECOVERY_SOURCE},prior_archive100_third_recovery=third)
        return first,second,third,fourth

    def test_original_f8_and_all_its_prior_fields_are_freshly_consumed(self):
        first,second,third,fourth=self.chain()
        with patch.object(lineage,'consume',return_value=third) as consume:
            self.assertEqual(reader.prior_archive100_evidence(fourth,reader.ARCHIVE100_FOURTH_RECOVERY_SOURCE),first)
            consume.assert_called_once_with(reader.ARCHIVE100_THIRD_RECOVERY_SOURCE,'recover_sigma3_archive100_third')
        for key in ('prior_archive100_third_recovery','prior_archive100_second_recovery','prior_archive100_recovery',
                    'prior_archive50_second_recovery','inherited_archive50','observer_recovery','legacy_archive5_recovery'):
            bad=copy.deepcopy(fourth);bad[key]={}
            with self.subTest(key=key),patch.object(lineage,'consume',return_value=third),self.assertRaises(ValueError):
                reader.prior_archive100_evidence(bad,reader.ARCHIVE100_FOURTH_RECOVERY_SOURCE)

    def test_holder_stdout_binds_all_three_prior100_proofs(self):
        pins=PINS[:-1]+(reader.ARCHIVE100_FOURTH_RECOVERY_SOURCE,);commit='a'*40
        _,admission,proof=output_fixture(commit);first,second,third,fourth=self.chain()
        for value in (admission,proof):
            value.update(recovery_commit=pins[0],observer_recovery_commit=pins[1],archive_recovery_commit=pins[2],archive100_recovery_commit=pins[3])
        admission.update(followthrough_source={'commit':pins[-1]},raw_source={'commit':pins[1]},
            archive5_recovery={'five':'09cc'},observer_recovery={'raw':'f979'},archive50_recovery={'fifty':'B894'},
            archive50_prior_recovery={'prior':'dc9'},archive100_recovery=fourth,archive100_prior_recovery=first,
            archive100_second_prior_recovery=second,archive100_third_prior_recovery=third)
        proof['followthrough_commit']=pins[-1];follow=proof['actual_followthrough_full_CPU_consumer']
        follow['source']=admission['followthrough_source']
        for key in ('archive5_recovery','observer_recovery','archive50_recovery','archive50_prior_recovery','archive100_recovery',
                    'archive100_prior_recovery','archive100_second_prior_recovery','archive100_third_prior_recovery'):
            follow[key]=admission[key]
        proof['independent_all_normal_return_raw']['source']=admission['raw_source']
        holder.accept_output(proof,commit,pins[-1],admission,*pins)
        for key in ('archive100_prior_recovery','archive100_second_prior_recovery','archive100_third_prior_recovery'):
            bad=copy.deepcopy(proof);bad['actual_followthrough_full_CPU_consumer'][key]={}
            with self.subTest(key=key),self.assertRaises(ValueError):holder.accept_output(bad,commit,pins[-1],admission,*pins)
        bad=copy.deepcopy(admission);bad.pop('archive100_third_prior_recovery')
        with self.assertRaisesRegex(ValueError,'authenticated F8'):holder.accept_output(proof,commit,pins[-1],bad,*pins)

    def test_missing200_still_refuses_before_any_source_or_spawn(self):
        with patch(__name__+'.PINS',PINS[:-1]+(reader.ARCHIVE100_FOURTH_RECOVERY_SOURCE,)):
            Exact100Route.test_less_than200_refuses_before_scope_source_or_spawn_on_new_route(self)
            InheritedHeldReceipts.test_exact_original5_and50_and_new100150200_native_sources_are_required(self)


if __name__=='__main__':unittest.main()
