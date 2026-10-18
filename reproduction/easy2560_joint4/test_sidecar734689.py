"""Tiny CPU fixtures for the rebound734689 sidecar; no SSH or credentials."""
import ast
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).parent/'code_backup/micro8-after731063-20261018/current-sidecar-archive734689.py'
SPEC = importlib.util.spec_from_file_location('sidecar734689',PATH)
module = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(module)


class Sidecar734689Tests(unittest.TestCase):
    def fixture(self):
        content = {'supplement.json':b'fixture supplement','complete.json':b'fixture complete',
                   'identity/accepted-archive.json':b'fixture accepted core'}
        source = module.plan(50)['source_directory']
        rows = [dict(relative_path=n,path=source+'/'+n,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                for n,raw in content.items()]
        value = dict(operator_file=module.CODE+'/reproduction/easy2560_joint4/checkpoint_evidence.py',
            operator_sha256=module.OPERATOR_SHA,verification=dict(source_base=module.BASE,
            source_commit=module.TRAINING,run_id=module.RUN,source_job=module.JOB,step=50,
            status='EASY2560_CHECKPOINT_EVIDENCE_SEALED',actual_cuda_restore_accepted=False,
            post_callback_rng_captured=False,single_original_job_prefix=False,
            inherited_history_assembled=True,files_to_archive=rows))
        return value,content

    def tar(self, content, *, duplicate=False):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            for name,raw in list(content.items()) + ([next(iter(content.items()))] if duplicate else []):
                member=tarfile.TarInfo(name); member.size=len(raw); archive.addfile(member,io.BytesIO(raw))
        stream.seek(0); return stream

    def test_default_main_never_reads_secret_or_network(self):
        output=io.StringIO()
        class NoSecret:
            def read(self,*a,**k):raise AssertionError('dry-run read a credential')
        with patch.object(module.sys,'argv',['sidecar','--step','50']),patch.object(module.sys,'stdin',NoSecret()), \
             patch.object(module.subprocess,'run',side_effect=AssertionError('dry-run SSH')), \
             patch.object(module.subprocess,'Popen',side_effect=AssertionError('dry-run SSH')), \
             patch.object(module.socket,'gethostname',side_effect=AssertionError('dry-run executed host check')),redirect_stdout(output):
            module.main()
        self.assertEqual(json.loads(output.getvalue())['status'],'DRY_RUN_NO_CREDENTIAL_READ_NO_SSH')

    def test_only_new_suffix_boundaries_and_exact_current_job_are_accepted(self):
        value,_=self.fixture(); files=module.validate_document(value,50); self.assertEqual(len(files),3)
        for step in (5,True,51,201):
            with self.subTest(step=step),self.assertRaises(ValueError):module.plan(step)
        for key,bad in (('source_job','730226'),('source_job','731063'),('source_commit','a61b48139386b346e4fef12511ece1ac0d09a8e3'),
                        ('source_base','/old/base'),('run_id','other'),('step',100),
                        ('single_original_job_prefix',True),('inherited_history_assembled',False),
                        ('actual_cuda_restore_accepted',True),('post_callback_rng_captured',True)):
            altered=copy.deepcopy(value); altered['verification'][key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):module.validate_document(altered,50)

    def test_path_escape_duplicates_and_wrong_verifier_are_rejected(self):
        value,_=self.fixture()
        for name in ('../escape','/escape','identity/../escape','sidecar-archive-receipt.json'):
            altered=copy.deepcopy(value); row=altered['verification']['files_to_archive'][0]
            row['relative_path']=name;row['path']=module.plan(50)['source_directory']+'/'+name
            with self.subTest(name=name),self.assertRaises(ValueError):module.validate_document(altered,50)
        altered=copy.deepcopy(value);altered['verification']['files_to_archive'].append(copy.deepcopy(altered['verification']['files_to_archive'][0]))
        with self.assertRaises(ValueError):module.validate_document(altered,50)
        altered=copy.deepcopy(value);altered['operator_sha256']='0'*64
        with self.assertRaises(ValueError):module.validate_document(altered,50)

    def test_exact_streamed_members_and_SHA_are_verified(self):
        value,content=self.fixture();files=module.validate_document(value,50)
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);module.receive_members(files,self.tar(content),stage)
            for name,raw in content.items():self.assertEqual((stage/name).read_bytes(),raw)
        for altered in (dict(content,**{'complete.json':b'wrong same bytes'}),
                        {k:v for k,v in content.items() if k!='complete.json'},dict(content,unknown=b'bad')):
            with tempfile.TemporaryDirectory() as directory,self.assertRaises(ValueError):
                module.receive_members(files,self.tar(altered),Path(directory))
        with tempfile.TemporaryDirectory() as directory,self.assertRaises(ValueError):
            module.receive_members(files,self.tar(content,duplicate=True),Path(directory))

    def test_remote_verification_pins_new_operator_and_precedes_import(self):
        program=module.remote_program(50);ast.parse(program)
        self.assertIn(module.OPERATOR_SHA,program);self.assertIn(module.CODE,program)
        self.assertLess(program.index('hashlib.sha256'),program.index('import checkpoint_evidence'))
        self.assertIn(module.plan(50)['source_directory'],program)


if __name__ == '__main__':unittest.main()
