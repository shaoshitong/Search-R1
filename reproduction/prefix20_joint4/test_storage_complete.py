import copy
from contextlib import ExitStack,redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.prefix20_joint4.assets import sha
from reproduction.prefix20_joint4 import storage_complete as module
from reproduction.prefix20_joint4 import storage_observe
from reproduction.prefix20_joint4.control import PERSONAL_ROOT,bind_repo


class StorageCompletionTests(unittest.TestCase):
    def fixture(self,base):
        prep=base/'preparation';prep.mkdir();denied=PERSONAL_ROOT+'/constructed/private'
        root=dict(role='prefix20_prelaunch_personal_storage_observation',account_experiment_root=PERSONAL_ROOT,
            returncode=1,timed_out=False,measurement_complete=False,checked_unix=time.time(),
            claim=dict(command=['du','-s','-B1',PERSONAL_ROOT]),
            stdout='1000\t'+PERSONAL_ROOT+'\n',stderr="du: cannot read directory '"+denied+"': Permission denied\n")
        for key in ('stdout','stderr'):
            p=prep/('root-'+key);p.write_text(root[key]);root[key+'_file']=str(p);root[key+'_sha256']=sha(p)
        root_path=prep/'root.json';root_path.write_text(json.dumps(root))
        part=dict(path=denied,command=['du','-s','-B1',denied],observer_uid=14090,returncode=0,
            before=dict(device=1,inode=2,uid=14090),after=dict(device=1,inode=2,uid=14090),
            stdout='200\t'+denied+'\n',stderr='')
        for key in ('stdout','stderr'):
            p=prep/('part-'+key);p.write_text(part[key]);part[key+'_file']=str(p);part[key+'_sha256']=sha(p)
        return dict(root,measurement_kind=module.KIND,measurement_complete=True,allocated_bytes_upper_bound=1200,
            aggregator_uid=14090,aggregator_source_sha256=sha(Path(module.__file__)),
            original_root_receipt_file=str(root_path),original_root_receipt_sha256=sha(root_path),
            permission_subtree_complements=[part])

    def test_full_permission_coverage_keeps_failed_root_and_counts_full_subtree(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();proof=self.fixture(base)
            self.assertEqual(module.validate_upper_bound(proof,base),1200)
            self.assertEqual(proof['returncode'],1)
            for field,value in (('allocated_bytes_upper_bound',1000),('returncode',0),('permission_subtree_complements',[])):
                bad=copy.deepcopy(proof);bad[field]=value
                with self.assertRaises(ValueError):module.validate_upper_bound(bad,base)
            part=proof['permission_subtree_complements'][0]
            Path(part['stdout_file']).write_text('300\t'+part['path']+'\n')
            with self.assertRaisesRegex(ValueError,'FILE_SHA'):module.validate_upper_bound(proof,base)

    def test_unrecognized_root_errors_and_symlink_partition_are_rejected(self):
        for text in ('du: I/O error\n',"du: cannot read directory '/etc/private': Permission denied\n"):
            with self.assertRaises(ValueError):module.denied_paths(text)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();target=root/'target';target.mkdir();link=root/'link';link.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'SYMLINK'):module.directory_identity(link)


class CanonicalRefreshTests(unittest.TestCase):
    """Real small receipts and atomic publication; only du/account identity is substituted."""
    fixture=StorageCompletionTests.fixture
    def setUp(self):
        self.saved_umask=os.umask(0o007)

    def tearDown(self):
        os.umask(self.saved_umask)

    def prepared(self,directory):
        base=Path(directory).resolve();old=self.fixture(base)
        canonical=base/'preparation/personal-storage-observation.json'
        canonical.write_text(json.dumps(old,indent=2))
        return base,canonical,canonical.read_bytes(),old

    def processes(self,base,canonical,old_bytes,*,part_error=False,root_complete=False):
        denied=PERSONAL_ROOT+'/constructed/private';commands=[]
        def popen(command,stdout,stderr):
            # Every actual external measurement observes the previous complete
            # canonical, including the denied-partition pass after raw publish.
            self.assertEqual(canonical.read_bytes(),old_bytes)
            commands.append(command)
            if command==['du','-s','-B1',PERSONAL_ROOT]:
                stdout.write(('1100\t'+PERSONAL_ROOT+'\n').encode())
                stderr.write(("du: cannot read directory '"+denied+"': Permission denied\n").encode() if not root_complete else b'')
                code=0 if root_complete else 1
            else:
                self.assertEqual(command,['du','-s','-B1',denied])
                stdout.write(('250\t'+denied+'\n').encode())
                stderr.write(b'du: I/O error\n' if part_error else b'')
                code=int(part_error)
            return SimpleNamespace(pid=12345,poll=lambda:code,wait=lambda **kwargs:code)
        stack=ExitStack()
        stack.enter_context(patch.object(storage_observe,'own_base',return_value=base))
        stack.enter_context(patch.object(module,'own_base',return_value=base))
        stack.enter_context(patch.object(storage_observe,'identity',return_value=dict(pid=12345,start_ticks=1,boot_id='test')))
        stack.enter_context(patch.object(module,'identity',return_value=dict(pid=12345,start_ticks=1,boot_id='test')))
        stack.enter_context(patch.object(module,'directory_identity',return_value=dict(device=1,inode=2,uid=14090)))
        stack.enter_context(patch.object(storage_observe.subprocess,'Popen',side_effect=popen))
        stack.enter_context(patch.object(storage_observe.os,'getuid',return_value=14090))
        stack.enter_context(patch.object(storage_observe.pwd,'getpwuid',return_value=SimpleNamespace(pw_name='jxu957')))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack,commands

    def test_refresh_keeps_complete_canonical_until_valid_aggregate_atomically_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            base,canonical,old_bytes,old=self.prepared(directory)
            stack,commands=self.processes(base,canonical,old_bytes)
            with stack:
                raw=storage_observe.observe(base,'a'*40,publish_canonical=False)
                self.assertFalse(raw['measurement_complete'])
                self.assertEqual(canonical.read_bytes(),old_bytes)
                raw_path=Path(raw['stdout_file']).parent/'result.json'
                raw_sha=sha(raw_path)
                self.assertEqual(json.loads(raw_path.read_text()),raw)
                publications=[];replace=os.replace
                def replace_complete(source,target):
                    if Path(target)==canonical:
                        self.assertEqual(canonical.read_bytes(),old_bytes)
                        pending=json.loads(Path(source).read_text())
                        self.assertEqual(module.validate_upper_bound(pending,base),1350)
                        publications.append(pending)
                    return replace(source,target)
                with patch.object(module.os,'replace',side_effect=replace_complete):
                    final=module.complete(base,raw_path,'a'*40)
            self.assertEqual(len(commands),2)
            self.assertEqual(publications,[final])
            self.assertEqual(json.loads(canonical.read_text()),final)
            self.assertEqual(module.validate_upper_bound(final,base),1350)
            self.assertEqual(sha(raw_path),raw_sha)
            self.assertEqual(sha(old['original_root_receipt_file']),old['original_root_receipt_sha256'])
            self.assertEqual(final['returncode'],1)
            self.assertEqual(final['original_root_receipt_sha256'],raw_sha)

    def test_failed_complement_preserves_previous_complete_canonical_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            base,canonical,old_bytes,old=self.prepared(directory)
            stack,_=self.processes(base,canonical,old_bytes,part_error=True)
            with stack:
                raw=storage_observe.observe(base,'a'*40,publish_canonical=False)
                raw_path=Path(raw['stdout_file']).parent/'result.json';raw_sha=sha(raw_path)
                with self.assertRaisesRegex(ValueError,'COMPLEMENT_FAILED'):
                    module.complete(base,raw_path,'a'*40)
            self.assertEqual(canonical.read_bytes(),old_bytes)
            self.assertEqual(module.validate_upper_bound(json.loads(canonical.read_text()),base),1200)
            self.assertEqual(sha(raw_path),raw_sha)
            self.assertEqual(len(list((base/'preparation').glob('storage-complements-*/0.result.json'))),1)

    def test_complete_root_observation_is_preserved_without_relabeling_or_interim_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            base,canonical,old_bytes,_=self.prepared(directory)
            stack,commands=self.processes(base,canonical,old_bytes,root_complete=True)
            with stack:
                raw=storage_observe.observe(base,'a'*40,publish_canonical=False)
                raw_path=Path(raw['stdout_file']).parent/'result.json'
                self.assertTrue(raw['measurement_complete'])
                self.assertEqual(raw['returncode'],0)
                # Permission completion remains explicitly limited to rc1 roots.
                with self.assertRaisesRegex(ValueError,'ORIGINAL_ROOT_DU_RECEIPT'):
                    module.complete(base,raw_path,'a'*40)
            self.assertEqual(canonical.read_bytes(),old_bytes)
            self.assertEqual(json.loads(raw_path.read_text()),raw)
            self.assertEqual(len(commands),1)

    def test_default_observer_still_publishes_canonical_for_existing_prelaunch_callers(self):
        with tempfile.TemporaryDirectory() as directory:
            base,canonical,old_bytes,_=self.prepared(directory)
            stack,_=self.processes(base,canonical,old_bytes)
            with stack:
                raw=storage_observe.observe(base,'a'*40)
            self.assertEqual(json.loads(canonical.read_text()),raw)

    def test_cli_no_publish_option_reaches_the_preserving_api(self):
        with patch('sys.argv',['storage_observe','--base','/source','--commit','a'*40,'--no-publish-canonical']),\
                patch.object(storage_observe,'observe') as observe:
            storage_observe.main()
        observe.assert_called_once_with('/source','a'*40,publish_canonical=False)


class PrelaunchSourcePromotionTests(unittest.TestCase):
    def source(self,base,commit,text):
        path=base/'source'/commit;path.mkdir(parents=True);(path/'entry.py').write_text(text)
        proof=dict(source_commit=commit,fresh_source_all_files_sha_verified=True,
            files={'entry.py':dict(bytes=len(text),sha256=sha(path/'entry.py'))})
        (path/'source-publication.json').write_text(json.dumps(proof))

    def test_only_unclaimed_exact_old_source_can_be_retired_and_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();(base/'preparation').mkdir();(base/'logs').mkdir()
            self.source(base,'a'*40,'old');self.source(base,'b'*40,'new')
            bind_repo(base,'a'*40)
            claim=base/'preparation/training-submit-claim.json';claim.write_text('{}')
            with self.assertRaisesRegex(ValueError,'CLAIMED'):bind_repo(base,'b'*40)
            self.assertEqual((base/'repo/entry.py').read_text(),'old');claim.unlink()
            bind_repo(base,'b'*40)
            self.assertEqual((base/'repo/entry.py').read_text(),'new')
            self.assertEqual((base/('repo.prelaunch-retired-'+'a'*40)/'entry.py').read_text(),'old')


if __name__=='__main__':unittest.main()
