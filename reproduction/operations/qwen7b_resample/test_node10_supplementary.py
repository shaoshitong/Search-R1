"""CPU regressions for GPU4 mapping, terminal anchor and once-only control."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import node10_supplementary_control as control

DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cpu_node10_supplementary', DIRECTORY / 'node10_supplementary_worker.py')
worker = importlib.util.module_from_spec(spec)
with patch.object(sys, 'path', [str(DIRECTORY)] + sys.path): spec.loader.exec_module(worker)


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value))


class Node10SupplementaryTests(unittest.TestCase):
    def fixture(self, base):
        run_id = control.original.RUN_ID
        owner = dict(pid=900, birth='111', unit='cpu-owner.service', mntns_inode=12, netns_inode=11,
                     cgroup='0::/system.slice/cpu-owner.service\n')
        slots = {name: dict(row, cuda_uuids=['GPU-%d' % i for i in row['indices']])
                 for name, row in control.queued.slots.specifications({}).items()}
        allocation = base / 'owner/allocation.json'
        put(allocation, dict(allocation_role='RETRIEVAL_RELAY_CPU_OWNER', role='downstream', run_id=run_id,
            owner=owner, slots=slots, gpu_lock_proofs={'GPU-%d' % i: dict(fd=i+30, device=1, inode=100+i) for i in range(8)}))
        ready = base / 'owner/owner-ready.json'
        put(ready, dict(status='RETRIEVAL_RELAY_OWNER_READY', role='downstream', run_id=run_id, owner=owner,
            relay=dict(owner, pid=901, birth='112'), source_owner=dict(pid=800, birth='90'),
            allocation_path=str(allocation), allocation_sha256=sha(allocation), slots=slots,
            job_ready_path=str(base / 'owner/job/ready.json'), consumer_deadline_epoch=50000))
        oldjob = base / 'pool/jobs/job-relay-slot-gpu1-tp1-3224037/attempt-original'
        oldallocation, failure = oldjob / 'allocation.json', oldjob / 'failure.json'
        put(oldallocation, dict(native_unit=worker.ANCHOR_UNIT, controller_pid=3224037,
            model_indices=[1], model_cuda_uuids=['GPU-1']))
        put(failure, dict(error_type='ValueError', error='Observed free VRAM cannot accommodate the configured model reserve'))
        intent = base / 'preparation/relay-generation-queued-owner-v2-slot-gpu1-tp1-r1/submit-intent.json'
        put(intent, dict(unit=worker.ANCHOR_UNIT, manifest=dict(unit=worker.ANCHOR_UNIT, role='slot', slot=worker.ANCHOR,
            instance='r1', run_id=run_id, generation='queued-owner-v2'),
            context=dict(downstream_owner_ready=str(ready), downstream_owner_ready_sha256=sha(ready))))
        terminal = base / 'terminal/complete.json'
        # Exact field names of the real owner-v2 spare-admission collector.
        put(terminal, dict(status=worker.TERMINAL_STATUS, node='h100-10', run_id=run_id, controlled_stop=False,
            release_observations=[dict(epoch=t, remaining_known_pids=[], remaining_original_births={},
                own_gpu_contexts=[], own_cgroups_empty=True) for t in (100, 104)],
            units=[dict(key='node10:gpu1-tp1', unit=worker.ANCHOR_UNIT, controller_pid=3224037,
                job_directory=str(oldjob), allocation_path=str(oldallocation), allocation_sha256=sha(oldallocation),
                failure_path=str(failure), failure_sha256=sha(failure))]))
        return SimpleNamespace(owner=owner, ready=ready, allocation=allocation, intent=intent, terminal=terminal,
            pool=SimpleNamespace(read=read, sha=sha, RUN_ID=run_id))

    def test_real_bootstrap_gpu4_stable_identity_and_unchanged_owner_registry(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            with patch.object(control.original,'BASE',str(base)):
                identity,files,manifest=control.bundle()
                self.assertEqual((identity,files,manifest),control.bundle())
            self.assertEqual(manifest['supplementary_slots'],{'gpu4-tp1':worker.SPECIFICATION})
            self.assertEqual(manifest['finite_node10_slots'],control.queued.slots.specifications({}))
            self.assertNotIn('gpu4-tp1',manifest['finite_node10_slots'])
            self.assertEqual(manifest['anchor_unit'],worker.ANCHOR_UNIT)
            self.assertNotEqual(manifest['unit'],worker.ANCHOR_UNIT)
            self.assertEqual(manifest['slot_hard_seconds'],28800)
            ast.parse(control.remote_source())
            stage=base/'preparation'/manifest['directory']/identity;stage.mkdir(parents=True)
            for name,raw in files.items():
                (stage/name).write_bytes(raw)
                self.assertEqual(manifest['files'][name],dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            put(stage/'manifest.json',manifest)
            upstream=base/'upstream/ready.json';put(upstream,dict(transport_ready_path='retained-upstream.json'))
            ready=base/'owner/ready.json';put(ready,dict(slots={},owner=dict(unit='owner.service'),
                upstream_owner_ready=str(upstream),job_ready_path=str(base/'owner/job/ready.json'),consumer_deadline_epoch=50000))
            put(stage.parent/'submit-context.json',dict(downstream_owner_ready=str(ready),downstream_owner_ready_sha256=sha(ready)))
            unit=manifest['unit'];group='/system.slice/'+unit
            detail='Id=%s\nMainPID=123\nActiveState=active\nControlGroup=%s\nMemoryMax=103079215104\nActiveEnterTimestampMonotonic=1000000000\n'%(unit,group)
            original_text,original_stat=Path.read_text,os.stat
            def text(path,*a,**k):return '0::'+group+'\n' if str(path)=='/proc/self/cgroup' else original_text(path,*a,**k)
            def stat(path,*a,**k):
                path=str(path)
                if path in ('/proc/self/ns/mnt','/proc/1/ns/mnt'):return SimpleNamespace(st_ino=12 if '/self/' in path else 13)
                if path in ('/proc/self/ns/net','/proc/1/ns/net'):return SimpleNamespace(st_ino=11)
                return original_stat(path,*a,**k)
            namespace={'__file__':str(stage/'relay_generation_bootstrap.py')}
            try:
                with patch('os.geteuid',return_value=0),patch('os.getpid',return_value=123),patch('socket.gethostname',return_value='h100-10'), \
                     patch.object(Path,'read_text',text),patch('os.stat',side_effect=stat),patch('time.time',return_value=2000), \
                     patch('time.monotonic',return_value=1500),patch('subprocess.check_output',return_value=detail), \
                     patch('os.chdir'),patch('os.dup2'),patch('builtins.print'),patch('os.execvpe') as execute:
                    exec(compile(files['relay_generation_bootstrap.py'],namespace['__file__'],'exec'),namespace)
                _,argv,env=execute.call_args.args
                flag=lambda name:argv[argv.index(name)+1]
                self.assertTrue(argv[2].endswith('/node10_supplementary_worker.py'))
                self.assertEqual([flag(k)for k in ('--model-gpus','--tp','--model-max-seqs','--model-batch-tokens',
                    '--model-kv-block-cap','--model-reserve-gib','--batch-reserve-seconds')],['4','1','64','8192','4096','1.0','1200'])
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'4');self.assertEqual(env['SEARCH_G1_PHYSICAL_WORLD_SIZE'],'4')
                self.assertEqual(flag('--run-seconds'),'28210')
                self.assertEqual(env['Q7B_RELAY_SLOT_NAME'],'gpu4-tp1')
                self.assertEqual(env['Q7B_NODE10_SUPPLEMENT_MANIFEST_SHA256'],sha(stage/'manifest.json'))
                self.assertEqual(env['Q7B_NODE10_SUPPLEMENT_CONTEXT_SHA256'],sha(stage.parent/'submit-context.json'))
            finally:
                if 'log' in namespace:namespace['log'].close()

    def test_original_full_terminal_shape_and_exact_downstream_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);f=self.fixture(base)
            def bind():return worker.anchor_binding(f.pool,base,read(f.ready),sha(f.ready),str(f.intent),sha(f.intent),str(f.terminal),sha(f.terminal))
            with patch.object(worker,'anchor_state',return_value=dict(MainPID='0',ActiveState='failed')):
                fact=bind();self.assertEqual(fact['controller_pid'],3224037)
                self.assertEqual(read(fact['failure_path'])['error_type'],'ValueError')
                wrong=read(f.intent);wrong['context']['downstream_owner_ready_sha256']='0'*64;put(f.intent,wrong)
                with self.assertRaisesRegex(ValueError,'unchanged CPU owner'):bind()
            f=self.fixture(base)
            props='Id=%s\nLoadState=loaded\nMainPID=3224037\nActiveState=active\n'%worker.ANCHOR_UNIT
            with patch.object(worker.subprocess,'check_output',return_value=props):
                with self.assertRaisesRegex(ValueError,'not terminal'):bind()

    def test_admission_anchor_gpu1_lifecycle_but_only_gpu4_cuda_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);f=self.fixture(base);ownunit=control.bundle()[2]['unit']
            original_ready,original_allocation=f.ready.read_bytes(),f.allocation.read_bytes()
            evidence=dict(unit=ownunit,cgroup='/system.slice/'+ownunit,unit_hard_end_epoch=40000,
                original_cuda_visible_devices='4',resource_limits=dict(runtime_max_seconds=28800))
            actual=dict(birth='555',unit=ownunit,mntns_inode=12,netns_inode=11)
            ready=read(f.ready);borrowed=SimpleNamespace(proof=dict(pid=901,process_owned=False),poll=lambda:None)
            manifest=base/'stage/manifest.json';m=control.bundle()[2];m['files']={};put(manifest,m)
            with patch.object(worker,'anchor_state',return_value=dict(MainPID='0',ActiveState='failed')):
                fact=worker.anchor_binding(f.pool,base,ready,sha(f.ready),str(f.intent),sha(f.intent),str(f.terminal),sha(f.terminal))
            context=base/'stage/submit-context.json';put(context,dict(anchor_terminal=fact))
            args=SimpleNamespace(base=str(base),model_gpus='4',tp=1,model_max_seqs=64,model_batch_tokens=8192,
                model_kv_block_cap=4096,model_reserve_gib=1.0,batch_reserve_seconds=1200,run_seconds=28500,
                upstream_ready=str(f.ready),allocation_evidence=str(f.ready))
            env=dict(Q7B_RELAY_SLOT_NAME=worker.SLOT,Q7B_NODE10_SUPPLEMENT_MANIFEST=str(manifest),
                Q7B_NODE10_SUPPLEMENT_MANIFEST_SHA256=sha(manifest),Q7B_NODE10_SUPPLEMENT_CONTEXT=str(context),
                Q7B_NODE10_SUPPLEMENT_CONTEXT_SHA256=sha(context))
            held=set();leases=[]
            def lock(path):
                path=Path(path)
                if str(path)in held:return None
                path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
                held.add(str(path));leases.append((path,fd));return fd
            pool=SimpleNamespace(**vars(f.pool),try_lock=lock,nvml=lambda:{'GPU-%d'%i:dict(index=i,uuid='GPU-%d'%i)for i in range(8)})
            original_text=Path.read_text
            def text(path,*a,**k):return 'fixture-boot-id\n' if str(path)=='/proc/sys/kernel/random/boot_id' else original_text(path,*a,**k)
            with patch.dict(os.environ,env),patch('socket.gethostname',return_value='h100-10'),patch('time.time',return_value=1000), \
                 patch.object(Path,'read_text',text), \
                 patch.object(worker.slot_worker,'ready',return_value=(f.ready,ready,read(f.allocation))), \
                 patch.object(worker.slot_worker,'borrowed_relay',return_value=borrowed), \
                 patch.object(worker.relay_owner,'own_evidence',return_value=(evidence,actual)), \
                 patch.object(worker,'anchor_state',return_value=dict(MainPID='0',ActiveState='failed')), \
                 patch.object(worker.slot_worker.source_slot,'held_lock',side_effect=lambda p,o,v:v) as primary, \
                 patch.object(worker.subprocess,'check_output',return_value=json.dumps(dict(actual_cuda_runtime_uuids=['GPU-4']))) as probe:
                binding,locks=worker.admit(pool,args)
                try:
                    self.assertEqual(binding['model_indices'],[4]);self.assertEqual(binding['model_cuda_uuids'],['GPU-4'])
                    self.assertEqual(binding['controller_birth'],555);self.assertEqual(binding['original_cuda_visible_devices'],'4')
                    self.assertEqual(binding['source_model_deadline_epoch'],39955)
                    self.assertEqual([call.args[0].name for call in primary.call_args_list],['GPU-1.lock','GPU-4.lock'])
                    self.assertEqual(leases[0][0].parent.name,'gpu1-tp1');self.assertEqual(leases[0][0].name,'GPU-1.lock')
                    self.assertEqual(leases[1][0].parent.name,'node10-supplement-gpu4-tp1');self.assertEqual(leases[1][0].name,'GPU-4.lock')
                    self.assertEqual(probe.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'4')
                    # The unmodified owner sees its declared anchor busy.
                    import fcntl
                    for fd in locks:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    busy=worker.relay_owner.source.busy_slots(pool,base,dict(owner=f.owner,slots=ready['slots']))
                    self.assertEqual(busy,[dict(slot='gpu1-tp1',uuid='GPU-1')])
                    with self.assertRaisesRegex(ValueError,'occupied'):worker.admit(pool,args)
                    self.assertEqual(probe.call_count,1)
                finally:
                    for fd in locks:os.close(fd)
                # A failed CUDA mapping releases both newly acquired own FDs.
                held.clear();probe.return_value=json.dumps(dict(actual_cuda_runtime_uuids=['GPU-1']))
                with self.assertRaisesRegex(ValueError,'CUDA UUID'):worker.admit(pool,args)
                for _,fd in leases[-2:]:
                    with self.assertRaises(OSError):os.fstat(fd)
            self.assertEqual(f.ready.read_bytes(),original_ready);self.assertEqual(f.allocation.read_bytes(),original_allocation)

    def test_actual_main_assembles_stage_submit_once_and_retains_anchor_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);f=self.fixture(base);starts=[];loaded=False
            output=base/'local';(output/'work').mkdir(parents=True)
            runtime=SimpleNamespace(bound_read=lambda pool,path,digest:read(path)if sha(path)==digest else self.fail('SHA'),
                same_process=lambda value:value,check_terminal=lambda pool,path:None)
            def run(argv,**kwargs):
                nonlocal loaded
                if argv[0]=='unshare':
                    stream=io.StringIO()
                    with contextlib.redirect_stdout(stream):exec(compile(kwargs['input'],'actual-main-inner-REMOTE','exec'),{})
                    return SimpleNamespace(returncode=0,stdout=stream.getvalue(),stderr='')
                if argv[:2]==['systemctl','show']:
                    unit=argv[2];props=dict(Id=unit,LoadState='loaded'if loaded else'not-found',MainPID='123'if loaded else'0',ActiveState='active'if loaded else'inactive')
                    return SimpleNamespace(returncode=0,stdout='\n'.join(k+'='+v for k,v in props.items()),stderr='')
                self.assertEqual(argv[0],'systemd-run');starts.append(argv);loaded=True
                return SimpleNamespace(returncode=0,stdout='started',stderr='')
            def transport(script,node):
                self.assertEqual(node,'h100-10');stream=io.StringIO()
                with contextlib.redirect_stdout(stream):
                    try:exec(compile(script,'actual-main-transport','exec'),{})
                    except SystemExit as error:self.assertEqual(error.code,0)
                return SimpleNamespace(returncode=0,stdout=stream.getvalue(),stderr='')
            common=['--published-commit','a'*40,'--published-ref','refs/heads/fixture']
            anchorargs=['--downstream-owner-ready',str(f.ready),'--downstream-owner-ready-sha256',sha(f.ready),
                '--anchor-intent',str(f.intent),'--anchor-intent-sha256',sha(f.intent),
                '--anchor-terminal-proof',str(f.terminal),'--anchor-terminal-sha256',sha(f.terminal)]
            with patch.object(control.original,'BASE',str(base)),patch.object(control.original,'PHYSICAL',str(base)), \
                 patch.object(control.original,'ROOT',output),patch('os.geteuid',return_value=0), \
                 patch('socket.gethostname',return_value='h100-10'),patch('time.time',return_value=1000), \
                 patch('subprocess.run',side_effect=run),patch.object(control.original.old,'transport',side_effect=transport), \
                 patch.object(control.original.old,'verify_publication',return_value=dict(status='EXACT_SOURCE_COMMIT_REMOTE_SHA_VERIFIED')), \
                 patch.dict(sys.modules,{'queued_relay_owner':runtime,'node10_supplementary_worker':worker}), \
                 patch.object(worker,'anchor_state',return_value=dict(MainPID='0',ActiveState='failed')),contextlib.redirect_stdout(io.StringIO()):
                for action in ('stage','submit','submit'):
                    with patch.object(sys,'argv',['node10-supplement',action,'--intent-id','gpu4-supp-'+action,*common,*anchorargs]):control.main()
                identity,files,manifest=control.bundle()
            self.assertEqual(len(starts),1)
            command=starts[0];self.assertIn('--property=RuntimeMaxSec=28800',command)
            self.assertIn('--setenv=Q7B_NATIVE_UNIT_INSTANCE='+manifest['unit'],command)
            target=command[command.index('nsenter'):]
            self.assertEqual(target[:4],['nsenter','--target','900','--mount'])
            root=base/'preparation'/manifest['directory'];intent=read(root/'submit-intent.json')
            self.assertEqual(intent['intent_id'],'gpu4-supp-submit')
            self.assertEqual(intent['manifest']['slot'],'gpu4-tp1')
            self.assertEqual(intent['context']['anchor_terminal']['terminal_proof_sha256'],sha(f.terminal))
            self.assertEqual(intent['context']['downstream_owner_ready_sha256'],sha(f.ready))
            self.assertEqual(read(f.intent)['manifest']['slot'],'gpu1-tp1')


if __name__ == '__main__': unittest.main()
