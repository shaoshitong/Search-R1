"""RPC bounds must reach cluster config, worker environment and process birth."""
import ast
import os
from pathlib import Path
import unittest
from unittest import mock

from reproduction.hpc3_dynrpo_periodic import ray_threads


class RayThreadBoundsTests(unittest.TestCase):
    def test_host_cpu_count_does_not_control_rpc_pool_sizes(self):
        with mock.patch.dict(os.environ,{},clear=True),mock.patch.object(os,'cpu_count',return_value=192):
            options=ray_threads.init_options()
            self.assertEqual(options['_system_config'],dict(num_server_call_thread=2,
                gcs_server_rpc_server_thread_num=2,gcs_server_rpc_client_thread_num=2))
            self.assertEqual(ray_threads.verify_environment(),options['_system_config'])
            for key,value in ray_threads.environment().items():
                self.assertEqual(options['runtime_env']['env_vars'][key],value)
            self.assertNotIn('num_cpus',options);self.assertNotIn('num_gpus',options)
            os.environ['RAY_num_server_call_thread']='48'
            with self.assertRaises(ValueError):ray_threads.verify_environment()

    def test_shell_birth_and_both_actual_ray_entries_use_shared_bounds(self):
        directory=Path(ray_threads.__file__).parent
        shell=(directory/'run.sh').read_text()
        for key,value in ray_threads.environment().items():
            self.assertLess(shell.index(key+'='+value),shell.index('python -m unittest'))
        smoke=ast.parse((directory/'driver_smoke.py').read_text())
        body=next(node for node in smoke.body if isinstance(node,ast.FunctionDef) and node.name=='main').body
        install=next(index for index,node in enumerate(body) if 'thread_options = init_options()' in ast.unparse(node))
        ray_import=next(index for index,node in enumerate(body) if ast.unparse(node)=='import ray')
        self.assertLess(install,ray_import)
        main=(directory.parents[1]/'verl/trainer/main_ppo_eco.py').read_text()
        self.assertIn('ray_runtime=init_options()',main)
        self.assertIn('**ray_runtime, **local_resources)',main)


if __name__=='__main__':unittest.main()
