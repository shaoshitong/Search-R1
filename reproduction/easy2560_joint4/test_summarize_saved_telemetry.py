"""Tiny scalar-only saved telemetry fixtures; no tensors, models or live jobs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reproduction.easy2560_joint4 import summarize_saved_telemetry as core


class SavedTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.run_id = 'dfe9c94084bb48b2a9bb7b60f9d882da'
    def tearDown(self): self.temp.cleanup()
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else core.canonical(value))
    def rows(self, path, rows, tail=b''):
        raw=b''.join(core.canonical(row)+b'\n' for row in rows)+tail; self.write(path,raw); return raw

    def fixture(self, name='old', steps=range(1,2), *, restored=None, retry=False):
        base=self.root/name; job='730226' if restored is None else '731063'
        source='8a5bac94b8230a99b81977f5290589e3d3708dfe' if restored is None else 'a61b48139386b346e4fef12511ece1ac0d09a8e3'
        run=dict(run_id=self.run_id, source_commit=source, source_sha256={'source.py':'b'*64},
            easy2560_selection=True, training_world_size=4, actor_logical_micro_batch_per_rank=4,
            question_probe=None, data_selection_mode='mixed_first', all_correct_virtual_zero=False,
            selection_candidate_questions=2560, selection_selected_questions=512)
        if restored:
            run['full_state_restore']=dict(kind='easy2560_native4',completed_outer=5,source_base=str(restored),
                source_job='730226',source_commit='8a5bac94b8230a99b81977f5290589e3d3708dfe')
        self.write(base/'logs/run-manifest.json',run); self.write(base/'job.id',(job+'\n').encode())
        uuids=['GPU-'+str(rank) for rank in range(4)]
        self.write(base/'logs/native-allocation-admission.json',dict(accepted=True,run_id=self.run_id,
            slurm_job_id=job,actual_cuda_runtime_uuids=uuids))
        updates=[]; schedules=[[] for _ in range(4)]
        for step in steps:
            attempt=format(step,'032x')
            update=dict(run_id=self.run_id,step=step,attempt_id=attempt,attempt_index=step,
                optimizer_writes_before=(step-1)*20,optimizer_writes_after=step*20,actual_optimizer_writes=20,
                full_budget_complete=True,update_status='complete',metrics={'critic/rewards/mean':1.23456789})
            updates.append(update)
            groups=[]
            for index in range(2560):
                labels=([1,0,0,0,0] if index<300 else [1]*5 if index<512 else
                        [1,1,0,0,0] if index<712 else [0]*5)
                category='all_correct' if sum(labels)==5 else 'all_wrong' if sum(labels)==0 else 'mixed'
                groups.append(dict(ordinal=index,strict_correctness=labels,category=category,selected=index<512))
            audit=dict(step=step,source_commit=source,policy_snapshot_id=self.run_id+':'+str(step)+':policy-'+attempt,
                fixed_candidate_budget=True,responses_per_group=5,groups=groups,attempted_candidates=2560,
                selected_groups=512,selected_mixed=300,selected_all_correct_fallback=212,
                raw_generated_policy_tokens=12345*(2 if retry else 1),raw_actual_retrieval_calls=345*(2 if retry else 1),
                raw_generated_trajectories=12800*(2 if retry else 1),pool_retry=int(retry),
                discarded_insufficient_pools=[dict(raw_generated_trajectories=12800)] if retry else [],
                candidate_fill_seconds=987.654321,candidate_generation_timings={'generation':123.456789})
            self.write(base/'artifacts/mixed-selection'/(attempt+'.json'),audit)
            for rank in range(4):
                common=dict(run_id=self.run_id,slurm_job_id=job,rank=rank,world_size=4,logical_update=step,
                    attempt_id=attempt,attempt_index=step,writes_after=step*20)
                schedules[rank]+=[dict(common,kind='BEGIN',observed_unix=step*100+10+rank/10),
                                  dict(common,kind='COMPLETE',observed_unix=step*100+20+rank/10)]
        self.rows(base/'logs/actual-updates.jsonl',updates,tail=b'{"step":999')
        for rank in range(4):self.rows(base/('logs/scheduler-rank-%d.jsonl'%rank),schedules[rank])
        samples=[]; source_map=hashlib.sha256(core.canonical(run['source_sha256'])).hexdigest()
        for step in steps:
            for start,end,used in ((9,9.1,9999),(10,10.1,100),(20,20.2,300),(20.25,20.4,8888)):
                samples.append(dict(role='actual_common8_gpu_occupancy',run_id=self.run_id,slurm_job_id=job,
                    source_map_sha256=source_map,query_started_unix=step*100+start,unix=step*100+end,
                    gpus=[dict(uuid=uuid,used_mib=used+rank,total_mib=81559) for rank,uuid in enumerate(uuids)]))
        self.rows(base/('logs/gpu-samples-'+self.run_id+'.jsonl'),samples,tail=b'{"unfinished":')
        return base

    def test_selected_mixed_fraction_and_selected_vs_candidate_em_have_different_denominators(self):
        base=self.fixture(); result=core.summarize([base],1); row=result['outers'][0]
        self.assertEqual(row['selected_mixed_fraction'],300/512)
        self.assertEqual(row['selected_all_correct_fallback_fraction'],212/512)
        self.assertEqual(row['selected_training_em']['correct_trajectories'],1360)
        self.assertEqual(row['selected_training_em']['fraction'],1360/2560)
        self.assertEqual(row['successful_candidate_pool']['mixed_fraction'],500/2560)
        self.assertEqual(row['successful_candidate_pool']['strict_em'],1760/12800)
        self.assertEqual(row['selected_sequence_reward_mean'],1.23456789)
        self.assertFalse(result['actual200_acceptance_claimed'])

    def test_original1to5_and_current6to7_stay_distinct_and_sources_are_unchanged(self):
        old=self.fixture(steps=range(1,6)); new=self.fixture('new',range(6,8),restored=old)
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result=core.summarize([old,new],7)
        self.assertEqual(result['missing_outers'],[])
        self.assertEqual([r['step'] for r in result['outers']],list(range(1,8)))
        self.assertEqual([r['source_job'] for r in result['outers']],['730226']*5+['731063']*2)
        self.assertEqual(result['outers'][-1]['cumulative_retained_adam_writes_per_rank'],140)
        for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)
        with self.assertRaisesRegex(ValueError,'EXPLICIT_PARENT_SEGMENT_REQUIRED'):core.summarize([new],7)
        foreign=self.fixture('other',range(1,2));p=foreign/'logs/run-manifest.json'
        manifest=json.loads(p.read_bytes());manifest['run_id']='e'*32;self.write(p,manifest)
        with self.assertRaisesRegex(ValueError,'ALLOCATION_IDENTITY_DIFFERS'):core.summarize([old,foreign],7)

    def test_cp50_child_truncates_parent_failed_suffix_without_relabelling(self):
        old=self.fixture(steps=range(1,6))
        middle=self.fixture('middle',range(6,53),restored=old)
        third=self.fixture('third',range(51,54),restored=middle)
        path=third/'logs/run-manifest.json';run=json.loads(path.read_bytes())
        parent=json.loads((middle/'logs/run-manifest.json').read_bytes())
        run['full_state_restore'].update(completed_outer=50,source_job='731063',source_commit=parent['source_commit'])
        self.write(path,run)
        result=core.summarize([old,middle,third],53)
        self.assertEqual(result['missing_outers'],[])
        self.assertEqual([r['step'] for r in result['outers']],list(range(1,54)))
        self.assertEqual([r['source_base'] for r in result['outers']],
                         [str(old)]*5+[str(middle)]*45+[str(third)]*3)
        self.assertEqual(result['segments'][1]['completed_outers_in_report'],list(range(6,51)))

    def test_wrong_run_audit_job_gpu_source_map_and_rank_attempt_are_rejected(self):
        changes=(('logs/actual-updates.jsonl','run_id','e'*32),
                 ('artifacts/mixed-selection/'+format(1,'032x')+'.json','source_commit','f'*40),
                 ('logs/gpu-samples-'+self.run_id+'.jsonl','slurm_job_id','123'),
                 ('logs/gpu-samples-'+self.run_id+'.jsonl','source_map_sha256','f'*64),
                 ('logs/scheduler-rank-2.jsonl','slurm_job_id','123'))
        for index,(relative,key,value) in enumerate(changes):
            with self.subTest(relative=relative,key=key):
                base=self.fixture('bad'+str(index)); path=base/relative
                if path.suffix=='.jsonl':
                    rows,_=core.ledger(path); rows[0][key]=value; self.rows(path,rows)
                else:
                    doc=json.loads(path.read_bytes());doc[key]=value;self.write(path,doc)
                with self.assertRaises(ValueError):core.summarize([base],1)

    def test_actor_window_excludes_outside_gpu_peaks_and_partial_lines_are_prefix_only(self):
        base=self.fixture(); result=core.summarize([base],1); row=result['outers'][0]
        self.assertEqual(row['actor_window']['begin_unix'],110)
        self.assertEqual(row['actor_window']['complete_unix'],120.3)
        self.assertIsNone(row['actor_update_timer_seconds'])
        self.assertIsNone(row['exact_full_outer_walltime_seconds'])
        self.assertEqual(row['gpu_memory']['sample_count'],2)
        self.assertEqual([r['sampled_peak_used_mib'] for r in row['gpu_memory']['per_gpu']],[300,301,302,303])
        record=result['segments'][0]['evidence']['actual_updates_complete_prefix']
        raw=(base/'logs/actual-updates.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
        self.assertEqual((record['bytes'],record['sha256']),(len(prefix),hashlib.sha256(prefix).hexdigest()))
        self.assertFalse(record['whole_active_file_stability_claimed'])
        self.assertGreater(record['ignored_or_unread_suffix_bytes'],0)
        self.assertEqual(result['segments'][0]['evidence']['gpu_samples_complete_prefix']['ignored_or_unread_suffix_bytes'],len(b'{"unfinished":'))

    def test_retry_generated_work_is_already_cumulative_without_changing_success_pool_denominator(self):
        base=self.fixture(retry=True); row=core.summarize([base],1)['outers'][0]
        self.assertEqual(row['successful_candidate_pool']['groups'],2560)
        self.assertEqual(row['generated_work']['trajectories'],25600)
        self.assertEqual(row['generated_work']['kept_model_tokens'],24690)
        self.assertEqual(row['generated_work']['logical_retrieval_calls'],690)
        self.assertEqual(row['generated_work']['discarded_insufficient_pool_count'],1)
        self.assertIsNone(row['generated_work']['original_engine_generated_token_total'])
        self.assertIsNone(row['generated_work']['http_transport_attempt_count'])

    def test_cli_repeated_bases_emit_json_only_without_new_files_or_tensor_reads(self):
        old=self.fixture(steps=range(1,6));new=self.fixture('new',range(6,7),restored=old)
        command=[sys.executable,'-B','-m',core.__name__,'--base',str(old),'--base',str(new),'--through-outer','6']
        before=set(self.root.rglob('*')); completed=subprocess.run(command,capture_output=True,text=True,check=True)
        self.assertEqual(completed.stderr,'');self.assertEqual(json.loads(completed.stdout)['through_outer'],6)
        self.assertEqual(before,set(self.root.rglob('*')))

    def test_generic_zero_retained_write_diagnostic_keeps_twenty_opportunities_separate_from_parameter_updates(self):
        # Current production actual-updates requires >=1 retained write. This
        # deliberately broader diagnostic input tests faithful count reporting;
        # real zero-write attempts normally appear only in actual-attempts.
        base=self.fixture(steps=range(1,3)); path=base/'logs/actual-updates.jsonl'; updates,_=core.ledger(path)
        updates[0].update(actual_optimizer_writes=0,optimizer_writes_after=0,full_budget_complete=False,
            update_status='generic_zero_write_diagnostic',opportunities_processed=20,opportunities_unattempted=0)
        updates[0]['metrics'].update({'actor/update_opportunities_this_attempt':20,
                                     'actor/skipped_update_opportunities':20})
        updates[1].update(optimizer_writes_before=0,optimizer_writes_after=20)
        self.rows(path,updates)
        attempts=[]
        for index,update in enumerate(updates):
            attempts.append(dict(run_id=self.run_id,slurm_job_id='730226',attempt_id=update['attempt_id'],
                attempt_index=update['attempt_index'],logical_update=update['step'],
                optimizer_writes_this_attempt=update['actual_optimizer_writes'],
                optimizer_writes_before=update['optimizer_writes_before'],optimizer_writes_after=update['optimizer_writes_after'],
                optimizer_calls_this_attempt=7 if index==0 else 20,optimizer_rollbacks_this_attempt=7 if index==0 else 0))
        self.rows(base/'logs/actual-attempts.jsonl',attempts,tail=b'{"partial":')
        for rank in range(4):
            path=base/('logs/scheduler-rank-%d.jsonl'%rank);rows,_=core.ledger(path)
            for event in rows:
                event['writes_after']=0 if event['logical_update']==1 else 20
                if event['logical_update']==1 and event['kind']=='COMPLETE':event['kind']='ABORTED_ATTEMPT'
            self.rows(path,rows)
        result=core.summarize([base],2);zero,later=result['outers']
        self.assertEqual((zero['retained_adam_writes_per_rank'],zero['cumulative_retained_adam_writes_per_rank']),(0,0))
        self.assertEqual((zero['opportunities_processed'],zero['skipped_opportunities'],zero['opportunities_unattempted']),(20,20,0))
        self.assertEqual((zero['optimizer_calls_this_attempt'],zero['optimizer_rollbacks_this_attempt']),(7,7))
        self.assertFalse(zero['full_budget_complete']);self.assertEqual(later['cumulative_retained_adam_writes_per_rank'],20)
        self.assertEqual(zero['actor_window']['terminal_kind'],'ABORTED_ATTEMPT')
        self.assertIsNone(zero['actor_window']['complete_unix'])
        self.assertEqual(result['segments'][0]['evidence']['actual_attempts_complete_prefix']['ignored_or_unread_suffix_bytes'],len(b'{"partial":'))
        self.assertIn('chosen attempt only',zero['optimizer_call_counts_scope'])

    def test_missing_attempt_or_skip_counts_are_unavailable_and_never_invented_zero(self):
        base=self.fixture();row=core.summarize([base],1)['outers'][0]
        for field in ('opportunities_processed','skipped_opportunities','opportunities_unattempted',
                      'optimizer_calls_this_attempt','optimizer_rollbacks_this_attempt'):
            self.assertIsNone(row[field]);self.assertEqual(row['write_activity_sources'][field],'unavailable')
        self.assertFalse(row['matching_actual_attempt_available'])
        # Even a matching native attempt can lack optional old counter fields.
        updates,_=core.ledger(base/'logs/actual-updates.jsonl');u=updates[0]
        self.rows(base/'logs/actual-attempts.jsonl',[dict(run_id=self.run_id,slurm_job_id='730226',
            attempt_id=u['attempt_id'],attempt_index=u['attempt_index'],logical_update=1,
            optimizer_writes_this_attempt=20,optimizer_writes_before=0,optimizer_writes_after=20)])
        row=core.summarize([base],1)['outers'][0]
        self.assertTrue(row['matching_actual_attempt_available']);self.assertIsNone(row['optimizer_rollbacks_this_attempt'])


if __name__=='__main__':unittest.main()
