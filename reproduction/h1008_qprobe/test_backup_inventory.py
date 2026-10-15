import copy
import json
from pathlib import Path
import tempfile
import unittest
from reproduction.h1008_qprobe.backup_inventory import inventory,digest


class BackupInventoryTests(unittest.TestCase):
    def fixture(self, root):
        root=Path(root);base=root/'run';cp=base/'checkpoints/actor/global_step_5';pairdir=base/'artifacts/completed-pairs/global_step_5';cache=root/'cache';model=root/'official'
        for p in (cp,pairdir,cache,model,base/'repo',base/'data',base/'preparation'):p.mkdir(parents=True)
        def file(path,text='tiny fixture'):
            path.write_text(text);return dict(path=str(path),bytes=path.stat().st_size,full_read_sha256=digest(path))
        run=dict(run_id='fresh',training_world_size=8,source_commit='source',initial_model=str(model),
            initial_model_sha256={'model.txt':file(model/'model.txt')['full_read_sha256']},input_sha256={'train.txt':file(base/'data/train.txt')['full_read_sha256']},
            question_probe=dict(cache_dir=str(cache),probe_architecture='qwen2_decoder3_count6',exploration_questions=64,selected_questions=512))
        feature=dict(schema_version=2,spec_sha='feature-spec',count=169028,files={})
        for name in ('hidden.bin','positions.bin','initialization.pt'):
            row=file(cache/name);feature['files'][name]=dict(bytes=row['bytes'],sha256=row['full_read_sha256'])
        file(cache/'manifest.json',json.dumps(feature));file(base/'preparation/h1008-launch-config.json',json.dumps(dict(precomputed_question_features=dict(path=str(cache),manifest_sha256=digest(cache/'manifest.json')))))
        source=file(base/'repo/source.py');file(base/'preparation/source-files.json',json.dumps({'source.py':source['full_read_sha256']}))
        pair=dict(status='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS',schema_version=1,step=5,world_size=8,run_id='fresh',actor_snapshot_id='fresh:5',source_base=str(base),checkpoint=str(cp),feature_spec_sha='feature-spec',
            files=[file(cp/'complete.json'),file(cp/'adam.txt'),file(cp/'question-head.txt')],boundary_driver=file(pairdir/'boundary-driver.txt'),
            source_run_manifest=file(pairdir/'run-manifest.json',json.dumps(run)),reward_probe_manifest=file(pairdir/'reward-manifest.json'),reward_files=[file(pairdir/'reward-head.txt')],boundary_worker_files=[])
        for rank in range(8):pair['boundary_worker_files'].append(dict(file(pairdir/('rng'+str(rank))),rank=rank))
        path=pairdir/'source-receipt.json';path.write_text(json.dumps(pair));return path,pair
    def test_inventory_covers_full_pair_official_features_and_source_without_pass_claim(self):
        with tempfile.TemporaryDirectory() as root:
            path,pair=self.fixture(root);plan=inventory(path,digest(path),allowed_roots=(root,))
            roles={role for row in plan['files'] for role in row['roles']}
            self.assertTrue({'checkpoint','boundary_driver','boundary_worker_rng','reward_probe_or_encoder','frozen_token_features','official_actor_reference_initializer','canonical_input','frozen_source'}<=roles)
            self.assertEqual(sum('boundary_worker_rng' in row['roles'] for row in plan['files']),8)
            self.assertFalse(plan['independent_full_sha_performed']);self.assertFalse(plan['semantic_reload_performed'])
    def test_official_alias_requires_complete_source_and_target_sha(self):
        for corrupt in (False,True):
            with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory() as root:
                path,pair=self.fixture(root);base=Path(pair['source_base']);stage=base/'backup-dependencies/official-model-once-v1';(stage/'model').mkdir(parents=True)
                original=Path(root)/'official/model.txt';target=stage/'model/model.txt';target.write_bytes(original.read_bytes());sha=digest(original)
                code=base/'preparation/stage-official-backup-fixture.py';code.write_text('small worker fixture')
                (stage/'INTENT.json').write_text(json.dumps(dict(run_id='fresh',files_sha256={'model.txt':sha},worker_source_sha256=digest(code))))
                (stage/'LAUNCH.json').write_text('{}')
                row=dict(source_path=str(original),staged_path=str(target),bytes=target.stat().st_size,expected_sha256=sha,source_full_read_sha256=sha,staged_full_read_sha256='0'*64 if corrupt else sha)
                (stage/'RESULT.json').write_text(json.dumps(dict(status='SAME_HOST_OFFICIAL_MODEL_STAGE_FULL_SHA_PASS',run_id='fresh',source_model=str(original.parent),files=[row])))
                if corrupt:
                    with self.assertRaisesRegex(ValueError,'Staging full SHA'):inventory(path,digest(path),allowed_roots=(root,))
                else:
                    plan=inventory(path,digest(path),allowed_roots=(root,));self.assertEqual(plan['source_aliases'][0]['exported_source_path'],str(target.resolve()))
                    self.assertFalse(plan['independent_full_sha_performed'])

    def test_missing_boundary_or_changed_pinned_cache_and_extra_checkpoint_are_rejected(self):
        for change in ('boundary','cache','extra'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as root:
                path,pair=self.fixture(root)
                if change=='boundary':pair['boundary_worker_files'].pop();path.write_text(json.dumps(pair))
                elif change=='cache':(Path(root)/'cache/manifest.json').write_text((Path(root)/'cache/manifest.json').read_text()+' ')
                else:(Path(pair['checkpoint'])/'unexpected.txt').write_text('unbound')
                with self.assertRaises(ValueError):inventory(path,digest(path),allowed_roots=(root,))


if __name__=='__main__':unittest.main()
