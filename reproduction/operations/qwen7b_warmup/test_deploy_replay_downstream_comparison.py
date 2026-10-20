"""Offline deployment, sealed-metadata and submission-ambiguity regressions."""
import contextlib
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

from . import deploy_replay_downstream_comparison as deploy


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fp(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def execute(block,namespace):
    # Each real control block is a separate short-lived remote interpreter.
    try:
        exec(block,namespace)
    finally:
        guard=namespace.pop('guard',None)
        if guard is not None:
            guard.close()


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.parent=Path(self.temporary.name).resolve()
        self.base=self.parent/'new-isolated-comparison'
        self.q=self.parent/'Q2/qualification-output'
        self.values=dict(parent=str(self.parent),base=str(self.base),qualification=str(self.q),commit='a'*40,
            e1_receipt=str(self.parent/'original/probe/epoch-0001.json'),
            e3_receipt=str(self.parent/'balanced/probe/epoch-0003.json'),job_name=deploy.JOB_NAME,
            bundle_sha256='b'*64,bundle_bytes=1,chunks=1,source_manifest_sha256='c'*64,
            renderer_sha256=deploy.digest(inspect.getsource(deploy.render_sbatch).encode()))

    def tearDown(self):
        self.temporary.cleanup()

    def namespace(self):
        own={'inputs':self.values}
        exec(deploy.COMMON,own)
        own['verify_code']=lambda:None
        return own

    def fixture(self):
        self.base.mkdir()
        (self.base/'inputs').mkdir()
        rows=[dict(qid='q%02d'%qid,trajectory_id='q%02d-r%d'%(qid,response))
              for qid in range(24) for response in range(5)]
        selection=self.q.parent/'selection.json'
        write(selection,dict(trajectories=rows))
        manifest=self.q/'layer26/manifest.json'
        write(manifest,dict(status='synthetic_metadata_only'))
        write(self.q/'complete.json',dict(status='BOUNDED_HF_REPLAY_CAPTURE_AND_NATIVE_COMPARISON_COMPLETE',
            model_revision='a09a35458c702b33eeacc393d103063234e8bc28',trajectories=120,
            bit_exact_native_vllm_claimed=False,accepted_for_training=False,
            source_selection=fp(selection),feature_manifests={'26':fp(manifest)}))
        for name,epoch,key,steps in [('E1',1,'e1_receipt',10464),('E3',3,'e3_receipt',13100)]:
            receipt_path=Path(self.values[key])
            checkpoint=receipt_path.with_suffix('.pt')
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            # Small opaque bytes are intentional: CPU prepare must NOT torch.load.
            checkpoint.write_bytes(b'synthetic '+name.encode())
            write(receipt_path,dict(fp(checkpoint),epoch=epoch,cursor=0,steps=steps,
                completed_epoch=True,actor_updates=0,independent_cpu_reload=True))
        ns=self.namespace()
        with contextlib.redirect_stdout(io.StringIO()):
            execute(deploy.PREPARE,ns)
        return ns

    def test_rendered_resources_exports_and_no_dependency(self):
        script=deploy.render_sbatch(str(self.base),str(self.base/'code'),str(self.q),'a'*64,
                                    str(self.base/'inputs/spec.json'),'b'*64)
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
        setup='\n'.join(line for line in script.splitlines() if line.startswith(('CMP_','export ')))
        subprocess.run(['bash','-euc',setup+'\ntest "$QWEN7B_CODE_ROOT" = '+str(self.base/'code')],check=True)
        for exact in ('--partition=acd_u','--qos=normal','--gres=gpu:1','--cpus-per-task=8','--mem=64G','--time=00:30:00'):
            self.assertIn(exact,script)
        self.assertNotIn('afterok',script)
        self.assertNotIn('@@CMP_',script)

    def test_actual_q2_completion_required_before_target_creation(self):
        self.assertFalse(self.base.exists())
        with self.assertRaises(FileNotFoundError):
            execute(deploy.CHECK,self.namespace())
        self.assertFalse(self.base.exists())

    def test_prepare_binds_real_receipts_without_loading_checkpoint(self):
        self.fixture()
        spec=json.loads((self.base/'inputs/checkpoints.json').read_text())
        self.assertEqual([(c['name'],c['epoch'],c['steps']) for c in spec['checkpoints']],
                         [('E1',1,10464),('E3',3,13100)])
        self.assertEqual(spec['source_receipts']['E1'],fp(Path(self.values['e1_receipt'])))
        prepared=json.loads((self.base/'preparation.json').read_text())
        self.assertFalse(prepared['login_model_or_tensor_loads'])
        self.assertFalse(prepared['accepted_for_training'])
        self.assertTrue(prepared['full_checkpoint_SHA_recheck_deferred_to_GPU'])
        self.assertEqual((self.base/'inputs/source-E3-sealed-receipt.json').read_bytes(),
                         Path(self.values['e3_receipt']).read_bytes())

    def test_changed_sealed_receipt_is_rejected(self):
        self.fixture()
        p=Path(self.values['e3_receipt'])
        changed=json.loads(p.read_text());changed['steps']+=1;write(p,changed)
        with self.assertRaisesRegex(AssertionError,'Sealed epoch receipt changed'):
            execute(deploy.PREPARE,self.namespace())

    def test_not_complete_g5_fixture_is_rejected(self):
        self.fixture()
        p=self.q.parent/'selection.json'
        changed=json.loads(p.read_text());changed['trajectories'][0]['qid']='extra';write(p,changed)
        q=json.loads((self.q/'complete.json').read_text());q['source_selection']=fp(p);write(self.q/'complete.json',q)
        with self.assertRaises(AssertionError):
            execute(deploy.CHECK,self.namespace())

    def submit_namespace(self,unknown=False):
        ns=self.fixture()
        ns['command']=lambda args,timeout=25:('PENDING|JobHeldUser' if args[:3]==['squeue','-h','-j'] else '')
        calls=[]
        def run(args,**kwargs):
            calls.append(args)
            if unknown:
                raise subprocess.TimeoutExpired(args,25)
            return types.SimpleNamespace(returncode=0,stdout='12345\n',stderr='')
        ns['subprocess']=types.SimpleNamespace(run=run,TimeoutExpired=subprocess.TimeoutExpired,PIPE=subprocess.PIPE)
        # Execute the exact injected remote renderer, proving its globals exist.
        exec(inspect.getsource(deploy.render_sbatch),ns)
        return ns,calls

    def test_unknown_sbatch_outcome_preserved_and_never_repeated(self):
        ns,calls=self.submit_namespace(unknown=True)
        with self.assertRaises(subprocess.TimeoutExpired):
            execute(deploy.SUBMIT,ns)
        intent=json.loads((self.base/'submission-intent.json').read_text())
        self.assertEqual(intent['status'],'SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT')
        with self.assertRaisesRegex(AssertionError,'Prior submit outcome'):
            execute(deploy.SUBMIT,ns)
        self.assertEqual(len(calls),1)
        self.assertFalse((self.base/'job.id').exists())

    def test_successful_held_submit_records_identity_before_release_and_no_duplicate(self):
        ns,calls=self.submit_namespace()
        with contextlib.redirect_stdout(io.StringIO()):
            execute(deploy.SUBMIT,ns)
        submission=json.loads((self.base/'submission.json').read_text())
        self.assertEqual(submission['status'],'REPLAY_COMPARISON_JOB_RELEASED')
        self.assertEqual((self.base/'job.id').read_text(),'12345\n')
        with self.assertRaisesRegex(AssertionError,'Existing job identity'):
            execute(deploy.SUBMIT,ns)
        self.assertEqual(len(calls),1)


if __name__=='__main__':
    unittest.main()
