"""Local evidence fixtures for the single explicit Python3.8 recovery."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .test_post_callback_future import fixture_helper

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('recover_future_fixture',HERE/'recover_post_callback_future_hk.py')
recover = importlib.util.module_from_spec(spec); spec.loader.exec_module(recover)


class RecoveryFixture:
    def __init__(self, directory):
        root=Path(directory).resolve(); self.helper=fixture_helper(root)
        self.hk=root/'hk'; self.parent=self.hk/'post-callback-boundary-control/job-745371/future-50-200'
        self.core=self.hk/'checkpoints'/self.helper.RUN/'global_step_5'
        self.stage=self.core/'.post-callback-dependencies.partial-bbbd605edd41455a864b3c0a3a86e6a1'
        self.control=self.core.parent/('.post-callback-dependencies-'+self.core.name)
        self.stage.mkdir(parents=True); self.control.mkdir(); self.parent.mkdir(parents=True)
        self.birth=dict(pid=recover.OLD_PID,start_ticks=1038489500,uid=1002,
            boot_id='34b77aa6-ff04-435f-88be-ff4b0cc3d0ae',hostname='xielab3')
        identity=dict(base=str(self.helper.BASE),source_job=self.helper.JOB,run_id=self.helper.RUN,
            training_commit=self.helper.TRAINING,dependencies_commit=recover.OLD_COMMIT)
        self.claim=dict(identity,birth={k:self.birth[k] for k in ('pid','start_ticks','uid','boot_id')},
            requested_steps=[50,100,150,200],dependencies_sha256='fab9806d9fb38dd8a1c1ece00a2fd04bb5889a7ff74ea929d8e361339ee289fe')
        self.launch=dict(identity,birth=self.birth,launcher_commit=recover.OLD_COMMIT,root=str(self.parent),public_code={})
        checkout=HERE.parents[2]
        # These are public immutable code bytes, read from the already local Git
        # object database. No remote/Git fetch or credential access occurs.
        for name in ('archive_post_callback_future_hk.py','observe_post_callback_future_hk.py'):
            raw=subprocess.check_output(['git','-C',str(checkout),'show',recover.OLD_COMMIT+':reproduction/operations/easy7b_20261021/'+name])
            path=self.hk/'post-callback-future-launch-tools'/recover.OLD_COMMIT/name
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            self.launch['public_code'][name]=dict(path=str(path),bytes=len(raw),sha256=self.helper.sha(path))
        raw=subprocess.check_output(['git','-C',str(checkout),'show',recover.OLD_COMMIT+':reproduction/easy2560_joint4/post_callback_dependencies.py'])
        path=self.hk/'post-callback-dependency-tools'/recover.OLD_COMMIT/'post_callback_dependencies.py'
        path.parent.mkdir(parents=True);path.write_bytes(raw)
        self.put(self.parent/'claim.json',self.claim);self.put(self.parent/'launch-receipt.json',self.launch)
        self.failed=dict(status='FAILED',error_type='AttributeError',error=recover.OLD_ERROR,completed_steps=[],
            source_job=self.helper.JOB,training_failure_inferred=False)
        self.put(self.parent/'failure.json',self.failed)
        self.put(self.control/'accept-claim.json',dict(pid=recover.OLD_PID,stage=str(self.stage)))
        self.put(self.control/'accept-failure.json',dict(error_type='AttributeError',error=recover.OLD_ERROR))
        self.put(self.parent/'qualification-global_step_5/dependencies-transfer.json',dict(stage=str(self.stage),
            emit_returncode=0,source_emit_completed=True,status='EASY2560_POST_CALLBACK_DEPENDENCIES_TRANSFER_SHA_PASS',
            transfer_file_count=6,transfer_bytes=50580970))
        for relative in ('launch-claim.json','runner.stdout.log','completed-five-read-only.json',
            'qualification-global_step_5/claim.json','qualification-global_step_5/source-tools.json',
            'qualification-global_step_5/dependencies-source-snapshot.json','qualification-global_step_5/dependencies-emit.stderr.log'):
            self.put(self.parent/relative,{})
        for name in ('receive-claim.json','receive-result.json'):self.put(self.control/name,{})
        for name in ('source-snapshot.json','transfer-receipt.json'):self.put(self.stage/name,{})
        self.future=SimpleNamespace(frozen_helper=Mock(return_value=self.helper),process_observation=Mock(return_value={'state':'TERMINAL'}))

    @staticmethod
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

    def constants(self):
        return dict(HK=self.hk,PARENT=self.parent,ROOT=self.parent/'recovery-python38-r1',CORE=self.core,STAGE=self.stage,
            ACCEPT_CONTROL=self.control,TOOLS_ROOT=self.hk/'post-callback-future-recovery-tools',
            OLD_FAILURE_SHA=self.helper.sha(self.parent/'failure.json'),
            ACCEPT_FAILURE_SHA=self.helper.sha(self.control/'accept-failure.json'),
            ACCEPT_CLAIM_SHA=self.helper.sha(self.control/'accept-claim.json'))


class RecoveryTests(unittest.TestCase):
    def observe(self,fixture):
        with patch.multiple(recover,**fixture.constants()),patch.object(recover,'load_future',return_value=fixture.future), \
                patch.object(recover.socket,'gethostname',return_value='xielab3'),patch.object(recover.os,'getuid',return_value=1002):
            return recover.recovery_prerequisites()

    def test_exact_terminal_failure_preserves_every_old_file_without_writes(self):
        with tempfile.TemporaryDirectory() as name:
            f=RecoveryFixture(name)
            before={str(p):f.helper.sha(p) for p in f.hk.rglob('*') if p.is_file()}
            result=self.observe(f)
            self.assertTrue(result['observed_terminal']);self.assertFalse(result['source_ssh_called'])
            self.assertEqual(result['parent_birth'],f.birth)
            self.assertIn(str(f.control/'accept-failure.json'),result['preserved_files'])
            self.assertEqual(before,{str(p):f.helper.sha(p) for p in f.hk.rglob('*') if p.is_file()})

    def test_live_or_unknown_original_birth_is_never_recoverable(self):
        with tempfile.TemporaryDirectory() as name:
            f=RecoveryFixture(name)
            for state in ('LIVE','UNKNOWN'):
                f.future.process_observation.return_value={'state':state}
                with self.assertRaisesRegex(ValueError,'NOT_PROVED_TERMINAL'):self.observe(f)

    def test_other_failure_or_foreign_stage_is_rejected_even_with_matching_fresh_sha(self):
        with tempfile.TemporaryDirectory() as name:
            f=RecoveryFixture(name);wrong=copy.deepcopy(f.failed);wrong['error']='different exception'
            f.put(f.parent/'failure.json',wrong)
            with self.assertRaisesRegex(ValueError,'EXACT_ORIGINAL_PYTHON38_FAILURE'):self.observe(f)
            f.put(f.parent/'failure.json',f.failed)
            f.put(f.control/'accept-claim.json',dict(pid=recover.OLD_PID,stage=str(f.core/'different-stage')))
            with self.assertRaisesRegex(ValueError,'EXACT_RECEIVED_STAGE'):self.observe(f)

    def test_old_evidence_mutation_is_rejected_and_all_runtime_sources_parse_as_python38(self):
        with tempfile.TemporaryDirectory() as name:
            f=RecoveryFixture(name);result=self.observe(f)
            f.put(f.parent/'failure.json',{})
            with self.assertRaisesRegex(ValueError,'EVIDENCE_CHANGED'):recover.preserved_unchanged(f.helper,result['preserved_files'])
        for name in ('archive_post_callback_future_hk.py','recover_post_callback_future_hk.py'):
            ast.parse((HERE/name).read_text(),feature_version=(3,8))


if __name__=='__main__':unittest.main()
