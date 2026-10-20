"""Narrow synthetic static-fork/namespace/rendering tests; no remote/model IO."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from . import extra_g5_fork_prepare as fork


def fixture(root):
    parent=root/'parent';parent.mkdir();code=root/'collector-code'
    name='reproduction/operations/qwen7b_warmup/extra_g5_prepare.py'
    old=Path(__file__).with_name('extra_g5_prepare.py').read_bytes()
    fork.write_once(code/name,old)
    groups=[dict(qid='q%d'%i,question_hash='h%d'%i,split='train',trajectory_ids=['old-%d-%d'%(i,j) for j in range(5)],labels=[0,1,0,1,0]) for i in range(4)]
    qids=[g['qid'] for g in groups];cohort=root/'cohort.json';fork.json_once(cohort,dict(cohort_qids=qids))
    c=dict(role=fork.ROLE,model_revision=fork.REVISION,base=str(parent),original_base=str(root/'never-read-old-base'),
        run_id='accepted-extra5-run',original_run_id='original0-run',sampling_attempt_id='accepted-attempt5',
        original_sampling_attempt_id='original0-attempt',extra_response_offset=5,extra_response_indices=list(range(5,10)),
        warmup_only=True,actor_updates_authorized=False,actor_optimizer_updates=0,capture_prefix_tokens=20,
        feature_layer=26,question_max_tokens=273,feature_dimension=3584,extra_shards=4,gpus_per_sampling_job=2,
        evaluation_only=False,prefix_diversity_selection=False,prefix_diversity={},source_commit=fork.COLLECTOR_COMMIT,
        code_root=str(code),code_files_sha256={name:fork.sha(code/name)},train_questions=4,train_trajectories=20,
        matched_cohort_plan=fork.proof(cohort),seed=20261021)
    entries=[]
    for shard,g in enumerate(groups):
        row=dict(qid=g['qid'],question_hash=g['question_hash'],original_trajectory_ids=g['trajectory_ids'],initial_prompt_token_ids=[1,2,3])
        p=parent/'inputs'/('shard-%02d'%shard)/'batch-000000.json'
        fork.json_once(p,dict(shard=shard,split='train',rows=[row],model_revision=fork.REVISION,role=fork.ROLE))
        entries.append(dict(fork.proof(p),questions=1,shard=shard))
        fork.json_once(p.parent/'partition.json',dict(shard=shard,questions=1,trajectories=5))
    q=dict(groups=groups,ordered_qid_sha256=fork.canonical_sha(qids),validation_qids_excluded=True,final_qids_excluded=True,
        matched_cohort_plan=c['matched_cohort_plan'],original_native_seeds_by_qid={g['qid']:list(range(i*5,i*5+5)) for i,g in enumerate(groups)},
        original_batches=[dict(path='/MUST_NOT_READ/prior-accepted-full-rollout.jsonl.gz',bytes=999999999,sha256='a'*64)],
        original_input_batches=[],role=fork.ROLE,question_selection='same sealed matched cohort')
    fork.json_once(parent/'inputs/questions.json',q)
    m=dict(batches=entries,questions=4,trajectories=20,ordered_qid_sha256=fork.canonical_sha(qids),run_id=c['run_id'],response_indices=list(range(5,10)),originalG5_preserved=True,role=fork.ROLE,model_revision=fork.REVISION)
    fork.json_once(parent/'inputs/manifest.json',m);fork.json_once(parent/'config.json',c)
    fork.json_once(parent/'prepared.json',dict(source_commit=c['source_commit'],extra_run_id=c['run_id'],source_qid_plan_sha256=m['ordered_qid_sha256'],config=fork.proof(parent/'config.json'),input_manifest=fork.proof(parent/'inputs/manifest.json')))
    pins={n:fork.sha(parent/n) for n in ('config.json','prepared.json','inputs/manifest.json','inputs/questions.json')}
    return parent,c,pins


class ExtraG5ForkTests(unittest.TestCase):
    def test_same_inputs_new_run_slots_no_bulk_read_and_idempotent(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);parent,c,pins=fixture(root);before={str(p):fork.sha(p) for p in parent.rglob('*') if p.is_file()}
            with mock.patch.object(fork,'validate_destination'):
                first=fork.prepare(parent,root/'offset10',10,pins,expected_questions=4)
                same=fork.prepare(parent,root/'offset10',10,pins,expected_questions=4)
                second=fork.prepare(parent,root/'offset15',15,pins,expected_questions=4)
            self.assertEqual(first,same)
            self.assertNotEqual(first['extra_run_id'],second['extra_run_id'])
            for offset,result in ((10,first),(15,second)):
                base=Path(result['base']);new=fork.read(base/'config.json');fork.validate_config(new)
                self.assertNotEqual(new['run_id'],c['run_id']);self.assertNotEqual(new['sampling_attempt_id'],c['sampling_attempt_id'])
                self.assertEqual(new['extra_response_indices'],list(range(offset,offset+5)))
                self.assertEqual(new['source_commit'],c['source_commit']);self.assertEqual(new['code_root'],c['code_root'])
                self.assertEqual((base/'inputs/questions.json').read_bytes(),(parent/'inputs/questions.json').read_bytes())
                m=fork.read(base/'inputs/manifest.json');self.assertEqual(m['run_id'],new['run_id'])
                self.assertEqual(m['batches'][0]['sha256'],fork.read(parent/'inputs/manifest.json')['batches'][0]['sha256'])
                provenance=fork.read(base/'fork-provenance.json');self.assertFalse(provenance['original_bulk_payloads_freshly_read'])
                self.assertEqual(provenance['original_bulk_proofs_inherited_count'],1)
                self.assertEqual(provenance['seed_disjointness']['status'],'REQUIRED_AT_FUTURE_G20_MERGE_NOT_YET_VERIFIED')
            self.assertEqual(before,{str(p):fork.sha(p) for p in parent.rglob('*') if p.is_file()})

    def test_static_parent_sha_or_token_binding_change_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            parent,c,pins=fixture(Path(raw));path=parent/'inputs/shard-00/batch-000000.json'
            value=fork.read(path);value['rows'][0]['initial_prompt_token_ids']=[];path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):fork.inspect_parent(parent,pins,4)
            with self.assertRaises(ValueError):fork.inspect_parent(parent,dict(pins,**{'config.json':'0'*64}),4)

    def test_safe_low_priority_independent_array_render(self):
        for offset in (10,15):
            script=fork.render_sbatch(Path(fork.target(offset)),Path('/data/user/test/immutable-code'),offset)
            subprocess.run(['bash','-n'],input=script,text=True,check=True)
            self.assertIn('#SBATCH --qos=normal\n',script);self.assertIn('#SBATCH --partition=acd_u\n',script)
            self.assertIn('#SBATCH --array=0-3%2\n',script);self.assertIn('#SBATCH --gres=gpu:2\n',script)
            self.assertNotIn('--dependency',script);self.assertNotIn('emergency',script)
            self.assertIn('q7b-g5-m6528-o%d'%offset,script)
        with self.assertRaises(ValueError):fork.render_sbatch(Path('/remote/space path'),Path('/remote/code'),10)
        with self.assertRaises(ValueError):fork.target(5)

    def test_destination_and_control_command_guards(self):
        with self.assertRaises(ValueError):fork.validate_destination(Path('/remote/parent'),Path('/tmp/local-data'))
        from .deploy_extra_g5_forks import self_test
        result=self_test();self.assertTrue(all(size<16000 for size in result['command_bytes'].values()))

if __name__=='__main__':unittest.main()
