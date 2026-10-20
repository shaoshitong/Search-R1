"""Pure small CPU contract tests; no source checkpoint/feature/model reads."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import peer_newseed_fit_operation as op
from . import deploy_peer_newseed_train as deploy


def checkpoint(cursor=16):
    return dict(cursor=cursor,steps=10464+cursor,
        accepted_hashes={k:'a'*64 for k in ('raw','ema','adam','raw_buffers','ema_buffers','rng')})


class PeerDeploymentTests(unittest.TestCase):
    def test_render_and_actual_export_value_no_root_replacement(self):
        code=str(op.TARGET/'code-0123456789ab')
        script=op.render_sbatch(op.TARGET,code)
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
        setup='\n'.join(line for line in script.splitlines() if line.startswith('export '))
        result=subprocess.check_output(['bash','-euc',setup+'\nprintf "%s" "$QWEN7B_CODE_ROOT"'],text=True)
        self.assertEqual(result,code)
        self.assertIn('--action train --base '+shlex.quote(str(op.TARGET)),script)
        self.assertIn('#SBATCH --qos=normal',script);self.assertIn('#SBATCH --gres=gpu:1',script)
        self.assertNotIn('afterok',script);self.assertNotIn('emergency',script)
        with self.assertRaises(ValueError):op.render_sbatch('/tmp/foreign',code)

    def test_fixed_onepass_recipe_and_original_source_references(self):
        source={k:dict(path='/remote/'+k,sha256='a'*64) for k in
            ('source_checkpoint','source_config','source_bank','model_config')}
        source['source_checkpoint']['sha256']='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
        source['model_dir']='/remote/model';original=copy.deepcopy(source)
        c=op.make_config(source,dict(commit='b'*40,files={'model.py':dict(sha256='c'*64)}),
                         {'path':'/remote/train'},{'path':'/remote/plan'},[{'name':'sameHF'}],{'kind':'NQ_only_enrichment'})
        self.assertEqual(source,original)
        self.assertEqual((c['global_questions'],c['update_budget'],c['fit_questions'],c['fit_real_trajectories']),(32,256,8192,81920))
        self.assertEqual(c['source_checkpoint'],source['source_checkpoint'])
        self.assertEqual(c['grouping'],dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise'))
        self.assertEqual(c['recipe']['loss_kwargs'],dict(trajectory_weight=1.,three_event_weight=1.))
        self.assertFalse(c['optimizer_reset']);self.assertTrue(c['source_all_four_rank_RNG_preserved'])
        self.assertFalse(c['four_rank_bitexact_resume_claim'])

    def test_legacy_SHA_only_reference_adds_actual_bytes_without_changing_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old-sealed-source.json';path.write_text('{}')
            ref=dict(path=str(path),sha256='a'*64)
            with patch.object(op.raw,'checked',side_effect=lambda item:item) as check:
                result=op.checked(ref)
            self.assertEqual(result,dict(ref,bytes=2))
            self.assertEqual(ref,dict(path=str(path),sha256='a'*64))
            self.assertEqual(check.call_count,1)

    def test_exact_resume_and_evaluation_only_cursors(self):
        self.assertEqual(op.resume_decision(None),'INITIAL_SOURCE_FORK')
        self.assertEqual(op.resume_decision(checkpoint(),{'cursor':16,'steps':10480},{'next_cursor':16}),
                         'RESUME_EXACT_ACCEPTED_CURSOR')
        self.assertEqual(op.resume_decision(checkpoint(256),{'cursor':256,'steps':10720},{'next_cursor':256}),
                         'RESUME_FINAL_EVALUATION_ONLY')
        self.assertEqual(op.resume_decision(None,complete=dict(status=op.COMPLETE_STATUS,cursor=256,steps=10720)),
                         'COMPLETE_NO_SUBMISSION')

    def test_unsaved_update_or_missing_components_is_never_relabelled_resume(self):
        for progress,intent in [({'cursor':17,'steps':10481},None),(None,{'next_cursor':17})]:
            with self.assertRaises(ValueError):op.resume_decision(checkpoint(),progress,intent)
        with self.assertRaises(ValueError):op.resume_decision(None,intent={'next_cursor':1})
        bad=checkpoint();bad['steps']=0
        with self.assertRaises(ValueError):op.resume_decision(bad)
        bad=checkpoint();del bad['accepted_hashes']['rng']
        with self.assertRaises(ValueError):op.resume_decision(bad)

    def test_all_control_actions_compile_and_payloads_under16KB(self):
        result=deploy.self_test()
        self.assertFalse(result['remote_actions_executed'])
        self.assertLess(max(result['command_bytes'].values()),16000)
        self.assertGreater(result['files'],15)

    def test_missing_actual_data_cannot_make_complete_or_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            missing=str(Path(directory)/'no-complete.json')
            values=dict(base=deploy.TARGET,target=deploy.TARGET,job_name=deploy.JOB_NAME,
                raw_complete=missing,identity_plan=missing,feature_completes=[missing]*4,
                validation_complete=missing,diagnostic_complete=missing)
            out=io.StringIO()
            with contextlib.redirect_stdout(out):exec(compile(deploy.remote_source(values,'probe'),'gate','exec'),{})
            value=json.loads(out.getvalue())
            self.assertTrue(value['missing']);self.assertFalse(value['training_submitted'])
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_early_baseline_gate_does_not_wait_for_training_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'complete.json';path.write_text(json.dumps({'status':op.VALIDATION_STATUS}))
            values=dict(base=deploy.TARGET,target=deploy.TARGET,job_name=deploy.JOB_NAME,
                        baseline_only=True,validation_complete=str(path))
            out=io.StringIO()
            with contextlib.redirect_stdout(out):exec(compile(deploy.remote_source(values,'probe'),'baseline-gate','exec'),{})
            self.assertEqual(json.loads(out.getvalue())['status'],'ACTUAL_COMPLETION_FILES_PRESENT_FULL_METADATA_GATE_PENDING')

    def test_completed_job_purged_from_queue_is_not_a_live_job_failure(self):
        namespace={'inputs':dict(base=deploy.TARGET,target=deploy.TARGET)}
        exec(compile(deploy.COMMON,'common','exec'),namespace)
        result=SimpleNamespace(returncode=1,stdout='',stderr='slurm_load_jobs error: Invalid job id specified\n')
        with patch.object(namespace['subprocess'],'run',return_value=result):
            self.assertEqual(namespace['command'](['squeue','-j','1']),'')
            with self.assertRaises(AssertionError):namespace['command'](['sacct','-j','1'])

    def test_preparation_has_no_tensor_deserialization(self):
        tree=ast.parse(Path(op.__file__).read_text())
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in ('prepare','prepare_baseline','metadata_features'):
                calls=[x.func for x in ast.walk(node) if isinstance(x,ast.Call)]
                self.assertFalse(any(isinstance(x,ast.Attribute) and x.attr in ('load','from_pretrained') for x in calls))


if __name__=='__main__':unittest.main()
