"""Real GNU du coverage plus adversarial storage proof/child-lifecycle tests."""
from contextlib import ExitStack
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import owner_partition_storage as storage


def gnu_du():
    for command in ('gdu','du'):
        executable=shutil.which(command)
        if executable:
            result=subprocess.run([executable,'--version'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if result.returncode==0 and b'GNU coreutils' in result.stdout:return executable
    return None


GNU_DU=gnu_du()


class ContractTests(unittest.TestCase):
    def test_complete_root_set_is_required_not_partial_stdout_or_extra_roots(self):
        self.assertEqual(storage.parse_du('12\t/a\n0\t/b\n',['/a','/b']),{'/a':12,'/b':0})
        for stdout in ('12\t/a\n','12\t/a\n1\t/a\n','12\t/a\n0\t/b','12\t/a\n0\t/b\n0\t/c\n',
                       '-1\t/a\n0\t/b\n','12 /a\n0\t/b\n'):
            with self.subTest(stdout=stdout),self.assertRaises(ValueError):storage.parse_du(stdout,['/a','/b'])

    def test_failed_du_only_supplies_a_frontier_never_usage(self):
        path=str(storage.MEASURE)+'/qprobe-teacher-warmup200-periodic25-thread5-20261016-r13/artifacts/behavior-traces'
        roots,missing=storage.permission_roots("du: cannot read directory '"+path+"': Permission denied\n")
        self.assertEqual(roots,[Path(path)]);self.assertEqual(missing,[])
        for error in ("du: cannot read directory '/etc/shadow': Permission denied\n",
                      "du: cannot read directory '"+path+"': Permission denied\ndu: unexpected error\n",
                      "du: cannot read directory '"+path+"': Permission denied\n"*2):
            with self.subTest(error=error),self.assertRaises(ValueError):storage.permission_roots(error)

    def test_exact_excludes_reject_glob_newline_or_option_injection(self):
        self.assertEqual(storage.du_command(['/a'],['/a/b']),['du','-s','-B1','--exclude=/a/b','--','/a'])
        for path in ('/a/*','/a/?','/a/[x]','/a/b\nc'):
            with self.assertRaises(ValueError):storage.du_command(['/a'],[path])

    def test_stale_complete_partitions_cannot_be_admitted(self):
        request={'created_unix':1000};du={'completed_unix':1100};owner={'completed_unix':1101}
        self.assertTrue(storage.fresh(request,du,owner,1200))
        self.assertFalse(storage.fresh(request,du,owner,1701))
        self.assertFalse(storage.fresh(request,{'completed_unix':1800},{'completed_unix':1801},1901))
        self.assertFalse(storage.fresh(request,du,owner,1099))

    def test_private_root_identity_is_observed_without_chmod_and_historical_mode_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            root=Path(temporary).resolve();private=root/'private';private.mkdir(mode=0o700)
            stack.enter_context(patch.object(storage,'ALLOWED',re.compile(re.escape(str(root))+r'/private(?:/.*)?')))
            stack.enter_context(patch.object(storage,'MOU',os.getuid()));stack.enter_context(patch.object(storage,'GID',os.getgid()))
            before=private.stat();rows,missing=storage.frontier([private,private/'child'])
            self.assertEqual(missing,[]);self.assertEqual(len(rows),1);self.assertEqual(rows[0]['mode'],0o700)
            storage.validate_roots(rows);self.assertEqual(private.stat().st_ctime_ns,before.st_ctime_ns)
            with self.assertRaises(ValueError):storage.validate_roots(rows+rows)
            private.rmdir()
            storage.validate_roots(rows,current=False)  # Historical proof does not require a permanent tmp root.
            with self.assertRaises(FileNotFoundError):storage.validate_roots(rows)

    def test_retained_stream_symlink_and_same_size_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();path=root/'stream';path.write_bytes(b'original')
            self.assertEqual(storage.read_bytes(path,(os.getuid(),)),b'original')
            link=root/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):storage.read_bytes(link,(os.getuid(),))

    def test_owner_helper_source_parses_as_python36(self):
        ast.parse(Path(storage.__file__).read_text(),feature_version=(3,6))


class HeldOwnerProof(unittest.TestCase):
    """Mutation attacks on retained owner/child/DU/request byte conjunctions."""
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(storage,'MOU',os.getuid()))
        self.stack.enter_context(patch.object(storage,'GID',os.getgid()))
        self.stack.enter_context(patch.object(storage,'ALLOWED',re.compile(re.escape(str(self.root))+r'/private[12]')))
        self.paths=[self.root/'private1',self.root/'private2']
        for path in self.paths:path.mkdir(mode=0o700)
        self.request_path=self.root/'request.json';self.folder=self.root/'mou-control';self.folder.mkdir()
        self.source={'commit':'c'*40,'publication_path':'/fixture/publication','publication_sha256':'a'*64}
        self.request={'source':self.source,'roots':[storage.root_identity(p) for p in self.paths],
            'excluded_paths':list(map(str,self.paths)),'created_unix':10}
        self.request_path.write_text(json.dumps(self.request));self.proof={'path':str(self.request_path),
            'sha256':storage.digest(self.request_path),'bytes':self.request_path.stat().st_size}
        self.supervisor={'pid':2147482000,'start_ticks':10,'sid':2147482000,'uid':os.getuid()}
        self.task=dict(self.supervisor,pid=2147482001,start_ticks=11)
        self.du=dict(self.supervisor,pid=2147482002,start_ticks=12)
        self.token='d'*32;self.command=storage.du_command(self.paths)
        stdout=''.join('%d\t%s\n'%(n,p) for n,p in zip((32768,4096),self.paths))
        for stream,raw in (('stdout',stdout.encode()),('stderr',b'')):(self.folder/('mou-du-'+stream)).write_bytes(raw)
        result=dict(actual_parent=self.task,actual_du=self.du,actual_du_wait_exit_code=0,command=self.command,
            stdout=stdout,stderr='',stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
            stderr_sha256=hashlib.sha256(b'').hexdigest(),stdout_bytes=len(stdout.encode()),stderr_bytes=0,
            started_unix=12,completed_unix=13,uid=os.getuid(),training_actions=0,permission_changes=0)
        self.write('mou-du-result.json',result)
        self.write('mou-du-binding.json',{k:result[k] for k in ('actual_parent','actual_du','command','started_unix')})
        self.write('claim.json',dict(role='read_only_MOU_partition_du',token=self.token,source=self.source,request=self.proof,
            command=storage.owner_command(sys.executable,'owner-supervise',self.request_path,'c'*40,self.token),
            training_actions=0,permission_changes=0))
        self.write('process.json',dict(actual_process=self.supervisor,token=self.token,source=self.source,request=self.proof))
        self.write('child.json',dict(actual_process=self.task,supervisor=self.supervisor,source=self.source,request=self.proof,
            token=self.token,command=storage.owner_command(sys.executable,'owner-worker',self.request_path,'c'*40,self.token)))
        self.write('worker-result.json',dict(actual_process=self.supervisor,actual_task_birth=self.task,token=self.token,
            request=self.proof,source=self.source,actual_task_wait_exit_code=0,training_actions=0,permission_changes=0,completed_unix=15))
        self.write('complete.json',dict(status='ACTUAL_MOU_SUBTREES_FULL_DU_HELD0',source=self.source,request=self.proof,
            actual_process=self.task,du_result_sha256=storage.digest(self.folder/'mou-du-result.json'),
            roots=self.request['roots'],root_identities_after=self.request['roots'],allocated_by_root={str(p):n for p,n in zip(self.paths,(32768,4096))},
            allocated_bytes=36864,private_modes_unchanged=True,training_actions=0,permission_changes=0,completed_unix=14))

    def write(self,name,value):(self.folder/name).write_text(json.dumps(value))

    def consume(self,**kwargs):return storage.consume_owner(self.request_path,self.request,self.proof,**kwargs)

    def alter(self,name,key,value):
        path=self.folder/name;content=json.loads(path.read_bytes());content[key]=value;self.write(name,content)

    def test_complete_exact_owner_root_set_and_actual_held_wait_accept(self):
        result=self.consume();self.assertEqual(result['allocated_bytes'],36864)
        self.assertEqual(result['actual_task'],self.task);self.assertEqual(result['actual_process'],self.supervisor)

    def test_self_reported_success_cannot_replace_a_real_failed_owner_child_wait(self):
        self.alter('worker-result.json','actual_task_wait_exit_code',1)
        with self.assertRaisesRegex(ValueError,'owner helper failed'):self.consume()

    def test_failed_du_and_partial_coverage_marker_never_supply_usage(self):
        self.alter('complete.json','allocated_by_root',{str(self.paths[0]):32768})
        with self.assertRaisesRegex(ValueError,'exact subtree coverage'):self.consume()

    def test_wrong_source_request_or_parent_cannot_be_inherited(self):
        for file,key,value in (('worker-result.json','source',{}),('child.json','request',{}),
                               ('child.json','supervisor',self.task)):
            original=(self.folder/file).read_bytes();self.alter(file,key,value)
            with self.subTest(file=file,key=key),self.assertRaises(ValueError):self.consume()
            (self.folder/file).write_bytes(original)

    def test_live_original_owner_task_is_a_wait_not_acceptance(self):
        with patch.object(storage,'live',side_effect=lambda p:p==self.task):
            with self.assertRaisesRegex(ValueError,'owner helper failed'):self.consume()
        with patch.object(storage,'live',side_effect=lambda p:p==self.supervisor):self.assertIsNone(self.consume())

    def test_historical_consumption_retains_before_after_without_requiring_permanent_tmp_roots(self):
        self.paths[0].rmdir();self.assertEqual(self.consume(current=False)['allocated_bytes'],36864)
        with self.assertRaises(FileNotFoundError):self.consume()
        self.alter('complete.json','root_identities_after',[])
        with self.assertRaisesRegex(ValueError,'exact subtree coverage'):self.consume(current=False)

    def test_dead_supervisor_without_wait_receipt_is_failure_not_an_infinite_wait(self):
        (self.folder/'worker-result.json').unlink()
        with self.assertRaisesRegex(ValueError,'dead without actual held wait'):self.consume()


@unittest.skipUnless(GNU_DU,'actual GNU coreutils du is required; enforced on published remote test run')
class ActualGNUPartitions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.tree=self.root/'experiments';self.tree.mkdir()
        self.private=self.tree/'experiment/artifacts/behavior-traces';self.private.mkdir(parents=True,mode=0o700)
        self.other=self.tree/'tmp/private';self.other.mkdir(parents=True,mode=0o700)
        (self.tree/'ordinary').write_bytes(b'a'*65536)
        (self.private/'trace').write_bytes(b't'*32768)
        (self.other/'buffer').write_bytes(b'b'*16384)
        (self.private/'empty').mkdir(mode=0o700)
        sparse=self.tree/'sparse'
        with sparse.open('wb') as stream:stream.truncate(64*2**20)
        outside=self.root/'outside';outside.write_bytes(b'x'*131072)
        (self.private/'external-link').symlink_to(outside)

    def du(self,paths,excluded=()):
        command=storage.du_command(paths,excluded);command[0]=GNU_DU
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,LC_ALL='C'))
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stderr,b'')
        return storage.parse_du(result.stdout.decode(),paths)

    def test_complement_plus_exact_private_roots_covers_all_directory_blocks_sparse_files_and_symlinks(self):
        roots=[self.private,self.other];before={p:stat.S_IMODE(p.stat().st_mode) for p in roots}
        full=self.du([self.tree])[str(self.tree)]
        complement=self.du([self.tree],roots)[str(self.tree)]
        owner=self.du(roots)
        self.assertEqual(complement+sum(owner.values()),full)
        # Some shared filesystems materialize a truncated sparse file. Compare
        # actual allocated blocks, rather than assuming support for holes.
        sparse=self.tree/'sparse'
        self.assertEqual(self.du([sparse])[str(sparse)],sparse.stat().st_blocks*512)
        self.assertGreaterEqual(full,sparse.stat().st_blocks*512)
        self.assertEqual({p:stat.S_IMODE(p.stat().st_mode) for p in roots},before)
        self.assertEqual(set(owner),set(map(str,roots)))

    def test_cross_partition_hardlink_can_only_increase_the_conservative_upper_bound(self):
        os.link(self.tree/'ordinary',self.private/'cross-partition-hardlink')
        full=self.du([self.tree])[str(self.tree)]
        upper=self.du([self.tree],[self.private,self.other])[str(self.tree)]+sum(self.du([self.private,self.other]).values())
        self.assertGreaterEqual(upper,full)
        self.assertEqual(upper-full,(self.tree/'ordinary').stat().st_blocks*512)


@unittest.skipUnless(Path('/proc/self/stat').exists(),'actual Linux process births are required')
class ActualHeldChildren(unittest.TestCase):
    def test_binding_publication_failure_still_reaps_the_real_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary).resolve();children=[];original=storage.subprocess.Popen
            def popen(*args,**kwargs):
                child=original(*args,**kwargs);children.append(child);return child
            with patch.object(storage.subprocess,'Popen',side_effect=popen),\
                 patch.object(storage,'publish',side_effect=OSError('actual binding publication failure')):
                with self.assertRaisesRegex(OSError,'publication failure'):
                    storage.run_du(folder,'binding-failed',[sys.executable,'-c','import time;time.sleep(.2)'],os.getuid())
            self.assertEqual(len(children),1);self.assertEqual(children[0].returncode,0)
            self.assertIsNone(storage.process(children[0].pid))

    def test_genuine_held0_and_retained_du_bytes_are_reconsumed_not_exit_markers(self):
        if not GNU_DU:self.skipTest('GNU du required')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();tree=root/'tree';tree.mkdir();(tree/'file').write_bytes(b'x'*32768)
            output=root/'output';output.mkdir();command=storage.du_command([tree]);command[0]=GNU_DU
            result=storage.run_du(output,'actual',command,os.getuid())
            accepted,proof=storage.consume_du(output,'actual',command,os.getuid())
            self.assertEqual(result,accepted);self.assertEqual(accepted['actual_du_wait_exit_code'],0)
            stream=output/'actual-stdout';raw=stream.read_bytes();stream.write_bytes(b'9'+raw[1:])
            with self.assertRaisesRegex(ValueError,'bytes differ|bytes changed'):storage.consume_du(output,'actual',command,os.getuid())


if __name__=='__main__':unittest.main()
