import copy
import json
import os
from pathlib import Path
import resource
import tempfile
import unittest

import raise_owned_nproc as core


def plan():
    return dict(schema_version=1,base=core.BASE,job=core.JOB,run_id=core.RUN,training_commit=core.COMMIT,
        target_soft_nproc=8192,node=core.NODE,boot_id='a'*8+'-'+ 'b'*4+'-'+ 'c'*4+'-'+ 'd'*4+'-'+ 'e'*12,
        operator_sha256='0'*64,operator_publication_commit='1'*40,protected_sha256={p:'2'*64 for p in core.PROTECTED},
        include_current_cpu_helper=True)


def identity(pid=100,start=900):
    root='/slurm/uid_14090/job_727251/step_batch'
    return dict(pid=pid,start_ticks=start,sid=100,uid=14090,uids=[14090]*4,state='S',name='ray::IDLE',threads=51,
        proc_device=42,proc_inode=pid+1000,
        hostname=core.NODE,boot_id=plan()['boot_id'],
        cgroup='12:cpuset:'+root+'\n11:freezer:'+root+'\n7:devices:'+root+'/task_0\n3:memory:'+root+'/task_0\n')


def helper(pid):
    value=identity(pid,999);value['cgroup']=value['cgroup'].replace('step_batch','step_3');return value


class Kernel:
    def __init__(self,limits):
        self.limits=dict(limits);self.limits.setdefault(os.getpid(),(2048,100000));self.sets=[];self.fail_pid=None
    def __call__(self,pid,kind,new=None):
        if kind!=resource.RLIMIT_NPROC:raise AssertionError('wrong resource')
        if pid not in self.limits:raise ProcessLookupError()
        old=self.limits[pid]
        if new is not None:
            if self.fail_pid==pid:raise PermissionError('fixture denied')
            self.sets.append((pid,tuple(new)));self.limits[pid]=tuple(new)
        return old


class NprocPromotionTests(unittest.TestCase):
    def test_kernel_changes_only_soft_and_preserves_actual_hard(self):
        row=identity();kernel=Kernel({100:(2048,100000)});events=[]
        value=core.promote(row,plan(),reader=lambda pid:copy.deepcopy(row),prlimit=kernel,journal=events.append)
        self.assertEqual(kernel.sets,[(100,(8192,100000))])
        self.assertEqual(value['before'],[2048,100000]);self.assertEqual(value['kernel_returned_old'],[2048,100000])
        self.assertEqual(value['after'],[8192,100000]);self.assertEqual(value['status'],'PROMOTED_SOFT_ONLY_VERIFIED')
        self.assertEqual(events[0]['status'],'PROMOTION_INTENT')

    def test_already_higher_or_unlimited_soft_never_lowered(self):
        for current in ((8192,100000),(16384,100000),(resource.RLIM_INFINITY,resource.RLIM_INFINITY)):
            with self.subTest(current=current):
                kernel=Kernel({100:current});row=identity()
                value=core.promote(row,plan(),reader=lambda pid:row,prlimit=kernel)
                self.assertFalse(value['changed']);self.assertEqual(kernel.sets,[]);self.assertEqual(kernel.limits[100],current)

    def test_too_small_hard_refuses_before_any_write(self):
        row=identity();kernel=Kernel({100:(2048,4096)})
        with self.assertRaisesRegex(ValueError,'HARD_LIMIT_BELOW'):
            core.promote(row,plan(),reader=lambda pid:row,prlimit=kernel)
        self.assertEqual(kernel.sets,[])

    def test_foreign_uid_job_step_birth_or_node_refuses_before_write(self):
        for key,value in [('uid',21930),('uids',[14090,14090,0,14090]),('start_ticks',901),('hostname','ACD1-26'),
                          ('cgroup',identity()['cgroup'].replace('job_727251','job_999999')),
                          ('cgroup',identity()['cgroup'].replace('step_batch','step_2'))]:
            with self.subTest(key=key,value=value):
                row=identity();actual=dict(row);actual[key]=value;kernel=Kernel({100:(2048,100000)})
                with self.assertRaisesRegex(ValueError,'BIRTH_OR_SCOPE'):
                    core.promote(row,plan(),reader=lambda pid:actual,prlimit=kernel)
                self.assertEqual(kernel.sets,[])

    def test_recheck_immediately_before_set_detects_pid_reuse(self):
        row=identity();calls=[row,identity(start=901)];kernel=Kernel({100:(2048,100000)})
        with self.assertRaisesRegex(ValueError,'BIRTH_OR_SCOPE'):
            core.promote(row,plan(),reader=lambda pid:calls.pop(0),prlimit=kernel)
        self.assertEqual(kernel.sets,[])

    def test_birth_change_during_durable_intent_is_rejected_before_set(self):
        row=identity();actual=dict(row);kernel=Kernel({100:(2048,100000)})
        def journal(value):
            if value['status']=='PROMOTION_INTENT':actual['start_ticks']+=1;actual['proc_inode']+=1
        with self.assertRaisesRegex(ValueError,'BIRTH_OR_SCOPE'):
            core.promote(row,plan(),reader=lambda pid:actual,prlimit=kernel,journal=journal)
        self.assertEqual(kernel.sets,[])

    def test_helper_scope_cannot_promote_another_pid_or_mismatched_cpu_steps(self):
        row=helper(100);kernel=Kernel({100:(2048,100000)})
        with self.assertRaisesRegex(ValueError,'BIRTH_OR_SCOPE'):
            core.promote(row,plan(),reader=lambda pid:row,prlimit=kernel,helper=True)
        self.assertEqual(kernel.sets,[])
        actual=helper(os.getpid());actual['cgroup']=actual['cgroup'].replace('11:freezer:/slurm/uid_14090/job_727251/step_3',
            '11:freezer:/slurm/uid_14090/job_727251/step_4')
        self.assertFalse(core.in_scope(actual['cgroup'],True))

    def test_real_kernel_return_mismatch_is_persisted_not_silently_accepted(self):
        row=identity();kernel=Kernel({100:(2048,100000)});events=[]
        def racing(pid,kind,new=None):
            if new is not None:kernel.limits[pid]=(4096,100000)
            return kernel(pid,kind,new)
        with self.assertRaisesRegex(ValueError,'CONCURRENT_LIMIT_WRITE'):
            core.promote(row,plan(),reader=lambda pid:row,prlimit=racing,journal=lambda value:events.append(copy.deepcopy(value)))
        self.assertEqual(events[-1]['status'],'POST_WRITE_UNVERIFIED_INSPECT_NO_RETRY')
        self.assertEqual(events[-1]['kernel_returned_old'],[4096,100000]);self.assertEqual(events[-1]['after'],[8192,100000])

    def test_exited_pid_is_recorded_without_any_limit_change(self):
        row=identity();kernel=Kernel({100:(2048,100000)})
        def gone(pid):raise ProcessLookupError()
        value=core.promote(row,plan(),reader=gone,prlimit=kernel)
        self.assertEqual(value['status'],'EXITED_BEFORE_ANY_WRITE');self.assertEqual(kernel.sets,[])

    def test_exact_cgroup_controller_partition_required(self):
        self.assertTrue(core.in_batch(identity()['cgroup']))
        for text in (identity()['cgroup'].replace('step_batch','step_3'),identity()['cgroup'].replace('uid_14090','uid_21930'),
                     identity()['cgroup'].replace('3:memory:', '3:wrong:'),identity()['cgroup']+'2:rdma:/slurm/uid_14090/job_999/step_batch\n'):
            self.assertFalse(core.in_batch(text))

    def test_two_passes_include_new_child_and_once_claim_prevents_second_apply(self):
        rows={100:identity(100),101:identity(101,901)};kernel=Kernel({100:(2048,100000),101:(2048,100000)})
        calls=0
        def collect(document):
            nonlocal calls;calls+=1
            # Initial + pass1-before + pass1-after omit a new child; pass2 sees it.
            return [copy.deepcopy(rows[100])] if calls<=3 else [copy.deepcopy(rows[100]),copy.deepcopy(rows[101])]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()/'rescue'
            value=core.execute(plan(),root=root,check=lambda p:{'actual':True},collect=collect,reader=lambda pid:copy.deepcopy(rows[pid]),prlimit=kernel,helper_reader=helper)
            self.assertEqual(value['status'],'SOFT_NPROC_PROMOTION_ACTUALLY_VERIFIED');self.assertEqual(value['passes_completed'],2)
            self.assertEqual(kernel.sets,[(os.getpid(),(8192,100000)),(100,(8192,100000)),(101,(8192,100000))])
            self.assertEqual(len(value['final_actual_kernel_readbacks']),2)
            again=core.execute(plan(),root=root,check=lambda p:(_ for _ in ()).throw(AssertionError('must not rerun')),prlimit=kernel)
            self.assertEqual(again['status'],'ALREADY_ATTEMPTED_NO_RETRY');self.assertEqual(len(kernel.sets),3)

    def test_partial_kernel_failure_preserves_changed_pid_and_rejects_retry(self):
        rows={100:identity(100),101:identity(101,901)};kernel=Kernel({100:(2048,100000),101:(2048,100000)});kernel.fail_pid=101
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()/'rescue'
            value=core.execute(plan(),root=root,check=lambda p:{},collect=lambda p:list(rows.values()),reader=lambda pid:rows[pid],prlimit=kernel,helper_reader=helper)
            self.assertEqual(value['status'],'PARTIAL_OR_FAILED_INSPECT_NO_RETRY')
            self.assertEqual([r['process']['pid'] for r in value['changed_processes']],[os.getpid(),100]);self.assertEqual(kernel.limits[100],(8192,100000))
            events=[json.loads(p.read_text()) for p in sorted(root.glob('kernel-event-*.json'))]
            self.assertTrue(any(e['status']=='KERNEL_SET_FAILED' and e['process']['pid']==101 for e in events))
            self.assertEqual(core.execute(plan(),root=root)['status'],'ALREADY_ATTEMPTED_NO_RETRY')

    def test_helper_is_raised_before_thread_growth_can_block_readonly_forks(self):
        rows={100:identity(100),101:identity(101,901)};kernel=Kernel({100:(2048,100000),101:(2048,100000)})
        uid_threads=[1995];forks=[]
        def check(document):
            if kernel.limits[os.getpid()][0]<uid_threads[0]:raise OSError('fixture fork: RLIMIT_NPROC exhausted')
            forks.append(uid_threads[0]);return {'actual_readonly_scontrol':True}
        def syscall(pid,kind,new=None):
            value=kernel(pid,kind,new)
            if pid==100 and new is not None:uid_threads[0]=2500
            return value
        with tempfile.TemporaryDirectory() as temporary:
            value=core.execute(plan(),root=Path(temporary).resolve()/'rescue',check=check,
                collect=lambda p:list(rows.values()),reader=lambda pid:rows[pid],prlimit=syscall,helper_reader=helper)
            self.assertEqual(value['status'],'SOFT_NPROC_PROMOTION_ACTUALLY_VERIFIED')
            self.assertEqual(kernel.sets[0],(os.getpid(),(8192,100000)))
            self.assertIn(2500,forks)

    def test_any_initial_hard_limit_failure_leaves_no_claim_and_no_kernel_write(self):
        rows={100:identity(100),101:identity(101,901)};kernel=Kernel({100:(2048,100000),101:(2048,4096)})
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()/'rescue'
            with self.assertRaisesRegex(ValueError,'HARD_LIMIT_BELOW'):
                core.execute(plan(),root=root,check=lambda p:{},collect=lambda p:list(rows.values()),reader=lambda pid:rows[pid],prlimit=kernel,helper_reader=helper)
            self.assertFalse((root/'claim.json').exists());self.assertEqual(kernel.sets,[])

    def test_plan_cannot_change_job_target_or_runtime(self):
        for key,value in [('job','999999'),('target_soft_nproc',100000),('training_commit','2'*40),('node','ACD1-26')]:
            document=plan();document[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):core.validate_plan(document)


if __name__=='__main__':unittest.main()
