"""Bounded local tests for read-only accepted-CP5 resume and private handoff."""
import ast
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from .test_post_callback_future import fixture_helper

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('accepted_resume_fixture',HERE/'resume_post_callback_future_hk.py')
resume=importlib.util.module_from_spec(spec);spec.loader.exec_module(resume)


class ResumeTests(unittest.TestCase):
    def test_prerequisite_requires_exact_r1_failure_terminal_birth_and_nested_accepted_receipt(self):
        with tempfile.TemporaryDirectory() as name:
            helper=fixture_helper(Path(name))
            birth=dict(pid=3867549,start_ticks=1038542290,uid=1002,
                boot_id='34b77aa6-ff04-435f-88be-ff4b0cc3d0ae',hostname='xielab3')
            identity=dict(base=str(helper.BASE),source_job=helper.JOB,run_id=helper.RUN,training_commit=helper.TRAINING)
            launch=dict(identity,birth=birth);claim=dict(identity,birth=birth)
            failure=dict(status='FAILED',error_type='ValueError',error=resume.ERROR,
                source_job=helper.JOB,training_failure_inferred=False)
            core=dict(base=str(helper.BASE),run_id=helper.RUN,source_commit=helper.TRAINING,step=5,core_archive=str(resume.CORE))
            receipt=dict(status='EASY2560_POST_CALLBACK_DEPENDENCIES_SHA_FSYNC_PASS',binding={'core':core},
                all_manifest_dependencies_covered=True,destination_fsynced=True,actual_cuda_restore_accepted=False,
                transfer_file_count=6,transfer_bytes=50580970)
            values={resume.FAILED/'launch-receipt.json':launch,resume.FAILED/'recovery-claim.json':claim,
                resume.FAILED/'failure.json':failure,
                resume.CORE/'post-callback-dependencies/dependency-archive-receipt.json':receipt}
            sha={resume.FAILED/'failure.json':resume.FAILURE_SHA,resume.FAILED/'recovery-failure.json':resume.FAILURE_SHA,
                resume.FAILED/'launch-receipt.json':resume.LAUNCH_SHA,
                resume.CORE/'post-callback-dependencies/dependency-archive-receipt.json':resume.DEPENDENCY_RECEIPT_SHA,
                resume.CORE.parent/('.post-callback-dependencies-'+resume.CORE.name)/'recover-python38-result.json':resume.RECOVERY_RESULT_SHA}
            helper.read_json=lambda path:values[Path(path)];helper.sha=lambda path:sha[Path(path)]
            future=SimpleNamespace(frozen_helper=Mock(return_value=helper),process_observation=Mock(return_value={'state':'TERMINAL'}))
            with patch.object(resume,'load_future',return_value=future),patch.object(resume,'preserve',return_value={'kept':{}}), \
                    patch.object(resume.socket,'gethostname',return_value='xielab3'),patch.object(resume.os,'getuid',return_value=1002):
                proof=resume.resume_prerequisites();self.assertFalse(proof['archive_writers_called'])
                self.assertTrue(proof['fresh_payload_inspect_pending'])
                future.process_observation.return_value={'state':'LIVE'}
                with self.assertRaisesRegex(ValueError,'NOT_PROVED_TERMINAL'):resume.resume_prerequisites()
                future.process_observation.return_value={'state':'TERMINAL'}
                receipt['binding']=core
                with self.assertRaisesRegex(ValueError,'ACTUALLY_ACCEPTED_CP5'):resume.resume_prerequisites()

    def test_private_normal_future_handoff_owns_new_root_without_archive_or_extra_process_calls(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);helper=fixture_helper(root);newroot=root/'r2';newroot.mkdir()
            done=dict(status='EASY7B_FUTURE_POST_CALLBACK_COMPANION_COMPLETE',completed_steps=[50,100,150,200])
            future=SimpleNamespace(ROOT=resume.ORIGINAL,frozen_helper=Mock(return_value=helper),
                validate_envelope=Mock(return_value=('fixture-secret',b'public',{})),
                current_birth=Mock(return_value={'pid':123}),run=Mock(return_value=done))
            envelope={'password':'fixture-secret'}
            with patch.object(resume,'ROOT',newroot),patch.object(resume,'load_future',return_value=future), \
                    patch.object(resume,'published_sources'),patch.object(resume,'unchanged') as unchanged, \
                    patch.object(resume,'resume_prerequisites',return_value={'preserved_files':{'old':{}}}):
                result=resume.run(envelope)
            self.assertEqual(future.ROOT,newroot);future.run.assert_called_once()
            self.assertEqual(result['completed_steps'],[50,100,150,200]);unchanged.assert_called_once()
            self.assertEqual(envelope,{})
            self.assertTrue((newroot/'resume-prerequisites.json').is_file())
        tree=ast.parse((HERE/'resume_post_callback_future_hk.py').read_text(),feature_version=(3,8))
        called={node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        self.assertFalse(called & {'recover_accept','snapshot','emit','receive','accept','Popen'})


if __name__=='__main__':unittest.main()
