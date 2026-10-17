"""CPU socket/lease fixtures for the observed concurrent TP1 initialization race."""
import errno
import multiprocessing
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from .vllm_ports import bounded_vllm_port_namespace, PortNamespaceExhausted


def installed_open_port_shape():
    # Same probe/release and unbounded-increment behavior as installed primary.
    port=int(os.environ['VLLM_PORT']) if 'VLLM_PORT' in os.environ else None
    while True:
        try:
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
                sock.bind(('',port or 0));return sock.getsockname()[1]
        except OSError:
            if port is None:raise
            port+=1


def fake_modules():
    utils=ModuleType('vllm.utils');utils.get_open_port=installed_open_port_shape
    executor=ModuleType('vllm.executor.gpu_executor');executor.get_open_port=utils.get_open_port
    return utils,executor,{'vllm.utils':utils,'vllm.executor.gpu_executor':executor}


def free_pool(width):
    for _ in range(100):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
            sock.bind(('',0));start=sock.getsockname()[1]
        if start+width>65535:continue
        held=[]
        try:
            for port in range(start,start+width):
                sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);held.append(sock);sock.bind(('',port))
            return start
        except OSError:
            pass
        finally:
            for sock in held:sock.close()
    raise RuntimeError('CPU fixture cannot find its tiny temporary TCP pool')


def competing_constructor(rank,root,start,barrier,queue):
    utils,executor,modules=fake_modules();held=[]
    try:
        with patch.dict(sys.modules,modules),bounded_vllm_port_namespace(
                dict(run_id='same-run',execution_id='same-execution'),rank,outer=5,phase='continue',
                utils_module=utils,pool_start=start,span=4,slots=8,lock_root=root) as namespace:
            ports=[utils.get_open_port(),executor.get_open_port()]
            barrier.wait(timeout=10)  # All four constructors remain concurrent.
            for port in ports:
                sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);sock.bind(('',port));sock.listen(1);held.append(sock)
            barrier.wait(timeout=10)
            queue.put(dict(rank=rank,ports=ports,receipt=namespace.receipt()))
    except BaseException as error:
        queue.put(dict(rank=rank,error=repr(error)))
        raise
    finally:
        for sock in held:sock.close()


class BoundedVllmPortTests(unittest.TestCase):
    def test_four_concurrent_constructors_probe_then_bind_unique_ports_without_serializing(self):
        start=free_pool(32);context=multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as temporary:
            barrier=context.Barrier(4);queue=context.Queue()
            processes=[context.Process(target=competing_constructor,args=(rank,temporary,start,barrier,queue))for rank in range(4)]
            for process in processes:process.start()
            try:
                rows=[queue.get(timeout=15)for _ in processes]
                for process in processes:process.join(timeout=10)
                self.assertEqual([process.exitcode for process in processes],[0]*4)
                self.assertFalse(any('error'in row for row in rows),rows)
                ports=[port for row in rows for port in row['ports']]
                self.assertEqual(len(set(ports)),8)
                ranges=[set(range(row['receipt']['port_start'],row['receipt']['port_end']+1))for row in rows]
                self.assertEqual(len(set.union(*ranges)),16)
                for row in rows:
                    self.assertTrue(row['receipt']['lease_held_until_process_exit'])
                    self.assertFalse(row['receipt']['model_initializations_serialized_across_processes'])
            finally:
                for process in processes:
                    if process.is_alive():process.terminate();process.join()
                queue.close();queue.join_thread()

    def test_range_exhaustion_never_overflows_into_the_next_rank(self):
        utils,executor,modules=fake_modules();start=free_pool(8);sockets=[]
        with tempfile.TemporaryDirectory() as temporary,patch.dict(sys.modules,modules):
            with bounded_vllm_port_namespace(dict(run_id='run'),0,outer=5,phase='continue',
                    utils_module=utils,pool_start=start,span=2,slots=4,lock_root=temporary) as namespace:
                try:
                    for port in range(namespace.start,namespace.end+1):
                        sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);sock.bind(('',port));sockets.append(sock)
                    # The original implementation would escape this rank's range.
                    self.assertEqual(installed_open_port_shape(),namespace.end+1)
                    with self.assertRaises(PortNamespaceExhausted):utils.get_open_port()
                    with self.assertRaises(PortNamespaceExhausted):executor.get_open_port()
                    self.assertEqual(namespace.ports,[])
                finally:
                    for sock in sockets:sock.close()
            namespace.close()

    def test_fd_lease_outlives_constructor_and_alias_env_rng_restore_on_error(self):
        utils,executor,modules=fake_modules();start=free_pool(32);config=dict(run_id='same',execution_id='same')
        with tempfile.TemporaryDirectory() as temporary,patch.dict(sys.modules,modules),patch.dict(os.environ,{'VLLM_PORT':'18000'}):
            rng=random.getstate()
            with bounded_vllm_port_namespace(config,0,outer=5,phase='continue',utils_module=utils,
                    pool_start=start,span=4,slots=8,lock_root=temporary) as first:
                self.assertEqual(executor.get_open_port(),first.initial_port)
                later=ModuleType('vllm.later');later.alias=utils.get_open_port;sys.modules['vllm.later']=later
            self.assertIs(utils.get_open_port,installed_open_port_shape)
            self.assertIs(executor.get_open_port,installed_open_port_shape)
            self.assertIs(later.alias,installed_open_port_shape)
            self.assertEqual(os.environ['VLLM_PORT'],'18000')
            try:
                with self.assertRaisesRegex(RuntimeError,'constructor failure'):
                    with bounded_vllm_port_namespace(config,0,outer=5,phase='continue',utils_module=utils,
                            pool_start=start,span=4,slots=8,lock_root=temporary) as second:
                        self.assertNotEqual(first.slot,second.slot)
                        raise RuntimeError('constructor failure')
                self.assertFalse(second.constructor_succeeded)
                self.assertTrue(first.constructor_succeeded)
                self.assertIs(utils.get_open_port,installed_open_port_shape)
                self.assertEqual(os.environ['VLLM_PORT'],'18000')
                self.assertEqual(random.getstate(),rng)
            finally:
                first.close();second.close();sys.modules.pop('vllm.later',None)


if __name__=='__main__':unittest.main()
