"""Small file fixtures for run binding and actual exported dependency aliases."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import checkpoint_backup_native as backup
from .checkpoint_backup_native_control import start_script
from .fixed_assets_backup_control import prepare_script, start_script as fixed_start_script

RUN = '6cd6f043ef1641ae91ac177b4c27fe4a'


class DynamicBackupTests(unittest.TestCase):
    def fixture(self, mount):
        base = Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-actorhidden20-new-' + RUN[:12])
        official = Path('/dataset/official')
        config = dict(base=str(base), run_id=RUN, actor_hidden_native=True,
            official_model_dir=str(official), model_files_sha256={})
        def write(path, value):
            actual = backup.exported(path); actual.parent.mkdir(parents=True, exist_ok=True)
            actual.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
            return actual
        def item(path, value):
            write(path, value)
            return dict(path=str(path), bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
        point=base/'outers/outer-000005'; state_dir=base/'actor/outer-000005'
        actor=item(state_dir/'rank0.pt', b'actor state')
        head=item(point/'probe-state-at-completion.pt', b'probe state')
        weights=b'official model'; model_sha=hashlib.sha256(weights).hexdigest()
        config['model_files_sha256']={'weights.bin':model_sha}
        alias=backup.OFFICIAL_STAGE/'model/weights.bin';write(alias,weights)
        row=dict(source_path=str(official/'weights.bin'),staged_path=str(alias),
            expected_sha256=model_sha,source_full_read_sha256=model_sha,staged_full_read_sha256=model_sha)
        write(backup.OFFICIAL_STAGE/'RESULT.json',dict(status='SAME_HOST_OFFICIAL_MODEL_STAGE_FULL_SHA_PASS',
            run_id='f'*32,source_model=str(official),files=[row]))
        write(backup.OFFICIAL_STAGE/'INTENT.json',dict(run_id='f'*32,files_sha256=config['model_files_sha256']))
        proof=dict(status=backup.CPU_PASS,run_id=RUN,outer=5,files=[dict(actor,independent_full_sha=True)],
            actor_actual_adam_steps_per_rank=[100]*8,probe_actual_adam_steps_per_rank=[0]*8,
            grounding_lifecycle=dict(assets=[dict(path=str(official/'weights.bin'),bytes=len(weights),
                sha256=model_sha,sha256_source='actual_saved_grounding_manifest_producer')]))
        joint=dict(run_id=RUN,outer=5,actor=dict(actual_adam_steps_per_rank=[100]*8,
            state_path=str(state_dir),scheduler_completed_outer=5),
            probe=dict(completed_outer=0,actual_adam_steps_per_rank=[0]*8,bytes=head['bytes'],sha256=head['sha256']),
            probe_snapshot=str(head['path']),pool_snapshot=str(point/'pool.json'))
        write(point/'complete.json',joint);write(point/'pool.json',{})
        write(point/'prefix/complete.json',{});write(point/'prefix/scan/scan-complete.json',dict(files={}))
        for path in ('probe/initialization-complete.json','preparation/source-files.json','question-pool/binding.json'):
            write(base/path,{})
        write(base/'preparation/launch-config.json',config)
        proof_path=base/'preparation/joint-native-0005-independent-full-sha-cpu.json'
        actual=write(proof_path,proof)
        return dict(run_id=RUN,outer=5,source_base=str(base),cpu_proof=str(proof_path),
            cpu_proof_sha256=backup.sha(actual)),official,alias

    def test_fresh_run_strictly_binds_proof_config_and_original_alias(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(backup,'MOUNT',Path(directory)):
            intent,official,alias=self.fixture(Path(directory))
            files,proof=backup.inventory(intent)
            mapped=next(row for row in files if row['original_path']==str(official/'weights.bin'))
            self.assertEqual(mapped['source_export_alias'],str(alias))
            self.assertEqual(mapped['source_path'],str(backup.exported(alias)))
            self.assertEqual(mapped['relative_path'],'external/dataset/official/weights.bin')
            self.assertIn('producer_receipt_not_independent',mapped['expected_sha_origin'])
            self.assertTrue(backup.checkpoint_namespace(RUN,5).name.startswith(RUN[:12]))
            wrong=dict(intent,run_id='a'*32)
            with self.assertRaisesRegex(ValueError,'fully read-back'):backup.inventory(wrong)

    def test_staged_alias_must_match_actual_official_identity(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(backup,'MOUNT',Path(directory)):
            intent,official,alias=self.fixture(Path(directory))
            config=backup.read(backup.exported(Path(intent['source_base'])/'preparation/launch-config.json'))
            bad=copy.deepcopy(config);bad['model_files_sha256']['weights.bin']='0'*64
            with self.assertRaisesRegex(ValueError,'exact model'):
                backup.exported_dependency(official/'weights.bin',bad)
            with self.assertRaisesRegex(ValueError,'Only pinned'):
                backup.exported_dependency('/dataset/some-other-run/weights.bin',config)

    def test_control_scripts_compile_for_explicit_new_run(self):
        namespace=str(backup.checkpoint_namespace(RUN,5))
        script=start_script(dict(namespace=namespace,run_id=RUN,outer=5),b'published worker','a'*40)
        compile(script,'native-copy-start','exec')
        self.assertIn(RUN,script)
        files={name:dict(code='',sha256='a'*64)for name in('fixed_assets_backup.py','checkpoint_backup_native.py')}
        compile(prepare_script('/mnt/data/shaoshitong/search-g1-prefix-selection200-actorhidden20-fresh', 'a'*40,files),'fixed-prepare','exec')
        compile(fixed_start_script('/mnt/data/inventory.json','a'*64,'b'*40,files,RUN),'fixed-start','exec')
        for value in ('', 'x'*32, RUN[:12], '../'+RUN):
            with self.assertRaises(ValueError):backup.run_identity(value)


if __name__=='__main__':unittest.main()
