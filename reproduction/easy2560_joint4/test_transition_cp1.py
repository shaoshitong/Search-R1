"""Exercise shell dispatch with CPU-only stub processes, never torch/CUDA/Ray."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ENTRY = Path(__file__).with_name('transition_cp1.sh')
STUB = '''#!{python}
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['TRANSITION_TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps(dict(args=args, pid=os.getpid(), cwd=os.getcwd(),
        world=os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE'),
        profile=os.environ.get('SEARCH_G1_EXECUTION_PROFILE'),
        source=os.environ.get('SEARCH_G1_SOURCE_COMMIT'),
        offline=os.environ.get('HF_HUB_OFFLINE'), omp=os.environ.get('OMP_NUM_THREADS'),
        https_proxy_present='HTTPS_PROXY' in os.environ)) + '\\n')
if args[:2] == ['-B', '-']:
    raise SystemExit(subprocess.run([sys.executable] + args, input=sys.stdin.read(), text=True).returncode)
module = args[args.index('-m')+1]
base = Path(os.environ['SEARCH_G1_BASE'])
if module == 'reproduction.easy2560_joint4.prepare_training':
    (base/'logs').mkdir(exist_ok=True)
    (base/'logs/run-manifest.json').write_text('{{"prepared": true}}')
if module == 'torch.distributed.run':
    if not (base/'logs/run-manifest.json').is_file():
        raise SystemExit(90)
    print('ACTUAL_STUB_NUMERICAL_WORKERS_TERMINATING', file=sys.stderr)
    raise SystemExit(int(os.environ['TRANSITION_TEST_NUMERICAL_EXIT']))
'''


class TransitionCP1Tests(unittest.TestCase):
    def execute(self, numerical_code):
        temporary = tempfile.TemporaryDirectory(prefix='cp1 shell fixture ')
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)/'run with spaces'
        (base/'env/bin').mkdir(parents=True)
        (base/'repo').mkdir()
        python = base/'env/bin/python'
        python.write_text(STUB.format(python=sys.executable))
        python.chmod(0o700)
        calls_path = Path(temporary.name)/'calls.jsonl'
        commit = 'a'*40
        values = [str(base), commit, '8', '/CP1 source/receipt.json',
                  'sparse-exact-v1', '/external receipt/receipt.json', '/source pool/0.pt']
        env = dict(os.environ, TRANSITION_TEST_CALLS=str(calls_path),
                   TRANSITION_TEST_NUMERICAL_EXIT=str(numerical_code),
                   HTTPS_PROXY='must-be-unset', SLURM_JOB_ID='123456')
        result = subprocess.run(['bash', str(ENTRY)] + values, env=env,
            capture_output=True, text=True, timeout=20)
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
        state = json.loads((base/'logs/cp1-numerical-transition.json').read_text())
        return base, result, calls, state, values

    def test_success_prepares_once_runs_four_isolated_ranks_then_fresh_production(self):
        base, result, calls, state, values = self.execute(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        modules = [row for row in calls if '-m' in row['args']]
        self.assertEqual([row['args'][row['args'].index('-m')+1] for row in modules], [
            'reproduction.easy2560_joint4.prepare_training', 'torch.distributed.run',
            'reproduction.easy2560_joint4.native_ray_resources'])
        prepare, numerical, production = [row['args'] for row in modules]
        self.assertEqual(prepare, ['-B', '-m', 'reproduction.easy2560_joint4.prepare_training',
            '--base', values[0], '--commit', values[1], '--actor-micro', '8',
            '--infra-profile', values[4], '--restore-receipt', values[3],
            '--external-retriever-receipt', values[5]])
        self.assertIn('--nproc_per_node=4', numerical)
        self.assertEqual(numerical[numerical.index('--module')+1],
                         'reproduction.easy2560_joint4.qualify_sparse_cuda')
        self.assertEqual(numerical[numerical.index('--raw-pool-file')+1], values[6])
        self.assertEqual(production, ['-B', '-m', 'reproduction.easy2560_joint4.native_ray_resources',
                                      '--base', values[0]])
        self.assertEqual(state['phase'], 'STARTING_FRESH_PRODUCTION')
        self.assertEqual(state['numerical_exit_code'], 0)
        self.assertFalse(state['numerical_workers_reused_for_production'])
        self.assertTrue(state['formal_full_state_restore_still_required'])
        for row in modules:
            self.assertEqual(Path(row['cwd']), (base/'repo').resolve())
            self.assertEqual((row['world'], row['profile'], row['source'], row['offline'], row['omp']),
                             ('4', 'native4_bf16_varlen_fp32_master', values[1], '1', '8'))
            self.assertFalse(row['https_proxy_present'])

    def test_numeric_rejection_preserves_log_exits_same_code_and_never_dispatches_production(self):
        base, result, calls, state, values = self.execute(37)
        self.assertEqual(result.returncode, 37)
        modules = [row['args'][row['args'].index('-m')+1] for row in calls if '-m' in row['args']]
        self.assertEqual(modules, ['reproduction.easy2560_joint4.prepare_training', 'torch.distributed.run'])
        self.assertEqual(state['phase'], 'NUMERICAL_DIAGNOSTIC_FAILED')
        self.assertEqual(state['numerical_exit_code'], 37)
        self.assertFalse(state['production_dispatch_allowed'])
        self.assertFalse(state['automatic_retry_or_relaxation'])
        self.assertIn('ACTUAL_STUB_NUMERICAL_WORKERS_TERMINATING',
                      (base/'logs/cp1-sparse-numerical-qualification.log').read_text())


if __name__ == '__main__':
    unittest.main()
