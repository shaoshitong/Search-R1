"""Small scalar JSON fixtures only; no remote routes, checkpoints or Torch."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .fresh_progress_summary_local import NAME, RUN_ID, RUNTIME, summarize


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, *, cost=True, pairs=True):
    out=root/'outputs'/NAME
    config=dict(run_id=RUN_ID,source_commit=RUNTIME,base='/mnt/data/current')
    config_sha=write(out/'CURRENT_CONFIG.json',config);facts={}
    for outer in range(1,77):
        version=RUN_ID+('-official0' if outer==1 else '-actor%04d'%(outer-1));roles={}
        for role in ('actor','probe'):
            rows=[]
            for rank in range(8):
                last=dict(rank=rank,outer=outer,actual_adam_steps=20*outer)
                if role=='actor':last.update(event='ACTUAL_ADAM_WRITE',writes_this_attempt=20)
                else:last['step']=20
                row=dict(rank=rank,writes=20,last=last,sha256=digest(f'{role}{outer}{rank}'),
                    path=('/mnt/data/original'if outer<=6 else'/mnt/data/current')+'/scalar-ledger')
                if role=='actor':row['restore']=dict(actor_version=version,actual_adam_before=20*(outer-1),outer=outer,rank=rank)
                else:row['first']=dict(step=1,actual_adam_steps=20*(outer-1)+1,outer=outer,rank=rank)
                rows.append(row)
            roles[role]=rows
        selection=dict(run_id=RUN_ID,outer=outer,actor_version=version,prefix_selection_sha256=digest('selection'+str(outer)))
        for role,count in (('continued1024',1024),('actor512',512)):
            selection[role]=dict(questions=count,mixed_questions=outer,mixed_fraction=outer/count,
                correct_count_histogram={str(k):(count-outer if k==0 else outer if k==1 else 0)for k in range(6)},
                mean_predicted_mixed_probability=outer/200)
        facts[str(outer)]=dict(outer=outer,joint_sha256=digest('joint'+str(outer)),actor_selection=selection,
            probe_metrics=dict(run_id=RUN_ID,outer=outer,actor_version=version,test={'balanced_accuracy':outer/200},
                checkpoint={'sha256':digest('probe'+str(outer))},own_em_sha256=digest('labels'+str(outer))),
            phase_timings={'actual-success':{'exit_code':0}},**roles)
    write(out/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json',facts)
    actual=dict(base=config['base'],config_sha256=config_sha,observed_unix=100.,
        latest_joint=dict(run_id=RUN_ID,source_commit=RUNTIME,outer=76,sha256=facts['76']['joint_sha256'],
            actor_adam=[1520]*8,probe_adam=[1520]*8),
        actor=dict(run_id=RUN_ID,completed_outer=77,actual_adam_steps_per_rank=[1540]*8),
        probe=dict(run_id=RUN_ID,completed_outer=76,actual_adam_steps_per_rank=[1520]*8),
        stage=dict(run_id=RUN_ID,outer=77,stage='probe-update'),native={'JobId':'3433'},actual_rank_update_ledgers=[])
    write(out/'LATEST_ACTUAL.json',actual)
    status=dict(run_id=RUN_ID,source_commit=RUNTIME,base=config['base'],config_sha256=config_sha,
        actual_complete_outer=77,actor_adam_per_rank=[1540]*8,probe_adam_per_rank=[1540]*8,
        goal_status='active',goal_complete=False,connection_evidence='preserve-original',custom={'untouched':True})
    write(out/'STATUS.json',status);write(root/'outputs/STATUS.json',dict(status,root_only='keep'))
    write(out/'GOAL_TURN_PROGRESS_CURRENT.json',{'goal_status':'active','goal_complete':False,'keep':'goal'})
    write(out/'REQUIREMENT_EVIDENCE_CURRENT.json',{'goal_complete':False,'requirements':[{'human_state':'keep'}]})
    mirrors=(root/'mirror1',root/'mirror2')
    for i,mirror in enumerate(mirrors):write(mirror/'STATUS.json',dict(status,mirror_only=i))
    for directory in (out,root/'outputs',*mirrors):(directory/'REPORT.md').write_text('historic report must survive\n')
    if cost:
        data={role:dict(run_id=RUN_ID,complete_outer_rows={})for role in ('generation','retrieval')}
        for outer in range(1,77):
            path=('/mnt/data/original'if outer<=6 else'/mnt/data/current')+'/outers/outer-%06d/complete.json'%outer
            for role,metric in (('generation','combined_actual_sampled_tokens'),('retrieval','total_distinct_logical_retrieval_calls')):
                data[role]['complete_outer_rows'][str(outer)]=dict(outer=outer,run_id=RUN_ID,
                    receipt_sha256={path:facts[str(outer)]['joint_sha256']},**{metric:100+outer})
        write(out/'COST_CURRENT_PROGRESS.json',data)
        write(out/'COST_COVERAGE_CURRENT.json',dict(sampled_policy_tokens_committed=sum(range(101,177)),logical_retrieval_calls_committed=sum(range(101,177))))
    if pairs:
        rows=[]
        for outer in range(1,77):
            rows.append(dict(outer=outer,run_id=RUN_ID,joint_sha256=facts[str(outer)]['joint_sha256'],
                actor_version=facts[str(outer)]['actor_selection']['actor_version'],questions=1024,trajectories=5120,
                same_batch_comparison=True,actual_probe_updates_per_rank=20,after_probe_checkpoint_sha256=digest('probe'+str(outer)),
                selection_file_sha256=digest('selection'+str(outer)),own_em_sha256=digest('labels'+str(outer)),
                before=dict(trajectory_bce=1.,group_bce=1.,total_loss=2.),after=dict(trajectory_bce=.9,group_bce=.8,total_loss=1.7),
                delta_after_minus_before=dict(trajectory_bce=-.1,group_bce=-.2,total_loss=-.3)))
        write(out/'PROBE_SAME_BATCH_LOSS_PAIRS.json',{'rows':rows})
    write(out/'RESOURCE_COST_CURRENT.json',dict(run_id=RUN_ID,actual_complete_outer=75,created_unix=90.,
        total_native_gpu_reservation_hours=12.5,cpu_accounting_complete=False,scope='partial reservation/lower bound'))
    return out,mirrors


def read(path):return json.loads(path.read_text())


class LocalSummaryTests(unittest.TestCase):
    def test_joint76_remains_complete_with_saved1540_and_both_mirrors_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();out,mirrors=fixture(root)
            summarize(root,'progress','fixture note',mirror_roots=mirrors)
            summary=read(out/'CURRENT_PROGRESS_SUMMARY.json')
            self.assertEqual(summary['actual_complete_outer'],76)
            self.assertEqual(summary['joint_adam_per_rank']['actor'],[1520]*8)
            self.assertEqual(summary['in_flight']['roles']['actor']['saved_adam_per_rank'],[1540]*8)
            self.assertEqual(summary['in_flight']['roles']['probe']['saved_adam_per_rank'],[1520]*8)
            optimization=read(out/'OPTIMIZATION_COST_CURRENT.json');self.assertEqual(optimization['roles']['actor']['synchronized_adam_updates'],1520)
            self.assertFalse(summary['resource_snapshots']['RESOURCE_COST_CURRENT.json']['boundary_matches'])
            for path in (out/'STATUS.json',root/'outputs/STATUS.json',*(p/'STATUS.json'for p in mirrors)):
                status=read(path);self.assertEqual(status['actual_complete_outer'],77)
                self.assertEqual(status['actor_adam_per_rank'],[1540]*8)
                self.assertEqual(status['connection_evidence'],'preserve-original')
                self.assertEqual(status['custom'],{'untouched':True});self.assertEqual(status['goal_status'],'active')
                self.assertEqual(status['derived_progress_summary']['actual_complete_outer'],76)
            self.assertEqual(read(root/'outputs/STATUS.json')['root_only'],'keep')
            for i,path in enumerate(mirrors):
                self.assertEqual(read(path/'STATUS.json')['mirror_only'],i)
                self.assertEqual((path/'CURRENT_PROGRESS_SUMMARY.json').read_bytes(),(out/'CURRENT_PROGRESS_SUMMARY.json').read_bytes())
            for p in (out,root/'outputs',*mirrors):self.assertEqual((p/'REPORT.md').read_text(),'historic report must survive\n')
            self.assertFalse(read(out/'GOAL_TURN_PROGRESS_CURRENT.json')['goal_complete'])

    def test_missing_cost_pairs_and_one_direct_role_remain_pending_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();out,mirrors=fixture(root,cost=False,pairs=False)
            facts=read(out/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json');del facts['76']['actor'];write(out/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json',facts)
            summarize(root,'verified_wait',mirror_roots=mirrors)
            s=read(out/'CURRENT_PROGRESS_SUMMARY.json')
            self.assertIsNone(s['cost_coverage']['generation']['total'])
            self.assertEqual(s['cost_coverage']['generation']['pending_outers'],list(range(1,77)))
            pair=read(out/'PROBE_PAIRED_LOSS_COMPONENT_SUMMARY.json');self.assertIsNone(pair['through_outer'])
            self.assertIsNone(pair['component_directions']['total_loss']['mean_delta'])
            opt=read(out/'OPTIMIZATION_COST_CURRENT.json');self.assertEqual(opt['roles']['actor']['through_outer'],75)
            self.assertEqual(opt['roles']['actor']['pending_outers'],[76]);self.assertEqual(s['actual_complete_outer'],76)

    def test_recent_five_is72_to76_not_last_single_round_bin(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();out,mirrors=fixture(root,cost=False,pairs=False)
            summarize(root,'progress',mirror_roots=mirrors)
            trend=read(out/'MIXED_SELECTION_TREND_CURRENT.json')
            self.assertEqual(trend['five_outer_means'][-1]['outer_count'],1)
            self.assertEqual(trend['recent5']['first_outer'],72);self.assertEqual(trend['recent5']['last_outer'],76)
            self.assertAlmostEqual(trend['recent5']['continued1024_mixed_mean'],74/1024)
            self.assertNotEqual(trend['recent5']['continued1024_mixed_mean'],trend['five_outer_means'][-1]['continued1024_mixed_mean'])

    def test_wrong_run_joint_or_pair_identity_is_rejected_without_any_write(self):
        for fault in ('run','joint','pair','pairsha','config'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();out,mirrors=fixture(root)
                if fault in ('pair','pairsha'):
                    p=out/'PROBE_SAME_BATCH_LOSS_PAIRS.json';data=read(p)
                    data['rows'][-1]['run_id'if fault=='pair'else'own_em_sha256']='foreign'
                elif fault=='config':
                    p=out/'LATEST_ACTUAL.json';data=read(p);data['config_sha256']=digest('foreign config')
                else:
                    p=out/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json';data=read(p)
                    if fault=='run':data['1']['actor_selection']['run_id']='foreign'
                    else:data['76']['joint_sha256']=digest('wrong joint')
                write(p,data);before={str(p.relative_to(root)):p.read_bytes()for p in root.rglob('*')if p.is_file()}
                with self.assertRaises(ValueError):summarize(root,'progress',mirror_roots=mirrors)
                after={str(p.relative_to(root)):p.read_bytes()for p in root.rglob('*')if p.is_file()}
                self.assertEqual(before,after)

    def test_missing_joint_does_not_promote_cached_facts_or_saved_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();out,mirrors=fixture(root)
            actual=read(out/'LATEST_ACTUAL.json');actual['latest_joint']=None;write(out/'LATEST_ACTUAL.json',actual)
            summarize(root,'verified_wait',mirror_roots=mirrors)
            self.assertIsNone(read(out/'CURRENT_PROGRESS_SUMMARY.json')['actual_complete_outer'])
            self.assertIsNone(read(out/'OPTIMIZATION_COST_CURRENT.json')['complete_outer'])
            self.assertTrue(read(out/'MIXED_SELECTION_TREND_CURRENT.json')['boundary_pending'])

    def test_successful_joint_retains_failed_attempt_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();out,mirrors=fixture(root,cost=False,pairs=False)
            path=out/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json';facts=read(path)
            facts['76']['phase_timings']['earlier-typed-oom']={'exit_code':42};write(path,facts)
            summarize(root,'progress',mirror_roots=mirrors)
            summary=read(out/'CURRENT_PROGRESS_SUMMARY.json')
            self.assertEqual(summary['actual_complete_outer'],76)
            self.assertFalse(summary['all_latest_phase_exits_zero'])
            self.assertTrue(summary['all8_actor_probe20_actual_writes'])

    def test_legacy_goal_boundary_sha_and_annotation_stay_consistent(self):
        for joint_available in (True,False):
            with self.subTest(joint_available=joint_available),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();out,mirrors=fixture(root,cost=False,pairs=False)
                path=out/'GOAL_TURN_PROGRESS_CURRENT.json';goal=read(path)
                old=dict(actual_complete_outer=75,latest_joint_sha256=digest('joint75'),
                    current_turn_verified_increment='old round75 note')
                goal.update(old);write(path,goal)
                if not joint_available:
                    actual=read(out/'LATEST_ACTUAL.json');actual['latest_joint']=None;write(out/'LATEST_ACTUAL.json',actual)
                summarize(root,'progress','new caller annotation',mirror_roots=mirrors)
                goal=read(path)
                self.assertEqual(goal['actual_complete_outer'],76 if joint_available else None)
                self.assertEqual(goal['latest_joint_sha256'],digest('joint76')if joint_available else None)
                self.assertEqual(goal['current_turn_verified_increment'],'new caller annotation')
                self.assertEqual(goal['current_turn_verified_increment_kind'],'caller_supplied_annotation_from_cli_note')
                self.assertEqual({key:goal['previous_turn_progress_annotation'][key]for key in old},old)
                self.assertEqual(goal['goal_status'],'active');self.assertFalse(goal['goal_complete'])
                self.assertEqual(goal['keep'],'goal')
                summarize(root,'progress','new caller annotation',mirror_roots=mirrors)
                self.assertEqual(read(path)['previous_turn_progress_annotation'],goal['previous_turn_progress_annotation'])

    def test_missing_pair_binding_sha_is_pending(self):
        for field in ('checkpoint','own_em_sha256','prefix_selection_sha256'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                root=Path(directory).resolve();out,mirrors=fixture(root,cost=False)
                path=out/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json';facts=read(path)
                container=facts['76']['actor_selection'if field=='prefix_selection_sha256'else'probe_metrics']
                del container[field];write(path,facts)
                summarize(root,'progress',mirror_roots=mirrors)
                pairs=read(out/'PROBE_PAIRED_LOSS_COMPONENT_SUMMARY.json')
                self.assertEqual(pairs['status'],'pending');self.assertEqual(pairs['through_outer'],75)
                self.assertEqual(pairs['pending_outers'],[76]);self.assertEqual(pairs['count'],75)
                self.assertIn('76',pairs['missing_binding_sha_by_outer'])


if __name__=='__main__':unittest.main()
