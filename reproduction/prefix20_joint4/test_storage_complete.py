import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from reproduction.prefix20_joint4.assets import sha
from reproduction.prefix20_joint4 import storage_complete as module
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
