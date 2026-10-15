"""Install/check four missing CPU calibration dependencies in this run's overlay.

Run on HPC3, after bootstrap and before submitting an independent namespace:
  SEARCH_G1_BASE=/data/user/jxu957/experiments/...-r4 \
    BASE/env/bin/python -m reproduction.hpc3_dynrpo_periodic.environment_overlay
Formal run.sh uses ``--check``; that mode never installs or changes packages.
The control Python supplies pip, while all imports/tests use BASE/env/bin/python.
No shared environment, NumPy, Torch, GPU, or model asset is modified.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import ast
import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = '/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016'
CONTROL = '/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin/python'
INDEX = 'http://harbor.internal.com:8081/repository/pypi-hkust/simple'
PINS = {'joblib': '1.3.2', 'scikit-learn': '1.3.2', 'scipy': '1.11.4', 'threadpoolctl': '3.2.0'}
MODULES = {'joblib': 'joblib', 'scikit-learn': 'sklearn', 'scipy': 'scipy', 'threadpoolctl': 'threadpoolctl'}
CALIBRATION = ('ts_probe_train.py', 'ts_probe_dump.py', 'probe_server.py', 'reproduction/common8/refit.py')


def owned_base(value):
    base = Path(value)
    if (sys.platform != 'linux' or re.fullmatch(re.escape(ROOT) + r'(?:-r[1-9][0-9]*)?', str(base)) is None
            or base.is_symlink() or base.resolve() != base or base.stat().st_uid != os.getuid()):
        raise ValueError('Overlay installation/check requires the owned remote HPC3 namespace')
    for part in ('env', 'preparation', 'tmp'):
        path = base / part
        if not path.is_dir() or path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError('Missing or foreign owned directory: ' + str(path))
    return base


def clean_env(base, overlay=None):
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower() and not k.startswith('PIP_')}
    env.update(SEARCH_G1_BASE=str(base), PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(base/'tmp'),
               PIP_CONFIG_FILE=os.devnull, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if overlay is not None:
        env['PYTHONPATH'] = str(overlay) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def pip_command(overlay):
    return [CONTROL, '-I', '-m', 'pip', '--disable-pip-version-check', 'install',
            '--index-url', INDEX, '--trusted-host', 'harbor.internal.com', '--no-deps',
            '--only-binary=:all:', '--no-cache-dir', '--no-compile', '--retries', '2', '--timeout', '30',
            '--target', str(overlay), *[name+'=='+version for name, version in PINS.items()]]


def assert_inactive(base, queue):
    # A historical run is never hot repaired, even if it appears to have zero updates.
    if (base/'logs/run-manifest.json').exists() or (base/'logs/training-launch-claim.json').exists():
        raise ValueError('Existing run state: create a new immutable recovery namespace before installing')
    for row in queue.splitlines():
        fields = row.split('|')
        if len(fields) != 3:
            raise ValueError('Cannot authenticate active Slurm queue rows')
        if fields[2].strip() == str(base):
            raise ValueError('Namespace already has an active Slurm job: ' + fields[0])


@contextlib.contextmanager
def installation_locks(base):
    with contextlib.ExitStack() as stack:
        for name in ('preparation/overlay-install.lock', 'preparation/preparation.lock',
                     'preparation/formal.lock', 'gpu-job.lock'):
            path = base/name
            if path.is_symlink():
                raise ValueError('Refuse a symlinked runtime lock')
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def inventory(overlay):
    allowed = re.compile(r'(?:joblib|sklearn|scipy|scipy\.libs|scikit_learn\.libs|threadpoolctl\.py|'
                         r'(?:joblib|scikit_learn|scipy|threadpoolctl)-[^/]+\.dist-info)')
    if overlay.is_symlink() or not overlay.is_dir():
        raise ValueError('Missing or symlinked CPU overlay')
    for item in overlay.iterdir():
        if allowed.fullmatch(item.name) is None and not (item.name == '__pycache__' and item.is_dir()):
            raise ValueError('Unexpected overlay content (NumPy/Torch overrides are forbidden): ' + item.name)
    result = {}
    for path in sorted(overlay.rglob('*')):
        if path.is_symlink():
            raise ValueError('Overlay wheel installed an untrusted symlink')
        if path.is_file():
            # A caller without -B may create normal interpreter caches after
            # acceptance. Ignore only direct, standard CPython cache files for
            # an existing regular source; all .py/.so/metadata bytes stay bound.
            cache = re.fullmatch(r'(.+)\.cpython-[0-9]+(?:\.opt-[12])?\.pyc', path.name)
            if path.parent.name == '__pycache__' and cache is not None:
                source = path.parent.parent/(cache.group(1)+'.py')
                if source.is_file() and not source.is_symlink():
                    continue
            h = hashlib.sha256()
            with path.open('rb') as f:
                for data in iter(lambda: f.read(4*1024**2), b''):
                    h.update(data)
            result[str(path.relative_to(overlay))] = h.hexdigest()
    if not result:
        raise ValueError('Empty package overlay')
    return result


def import_requests(repo):
    """Read actual top-level imports without executing refit's CLI/server setup."""
    requests = []
    for name in CALIBRATION:
        for node in ast.parse((repo/name).read_text(), filename=name).body:
            if isinstance(node, ast.Import):
                requests.extend((a.name, None) for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module != '__future__':
                if node.level:
                    raise ValueError('Unreviewed relative calibration import')
                requests.extend((node.module, a.name) for a in node.names)
    return sorted(set(requests), key=lambda row: (row[0], row[1] or ''))


def core_identity():
    result = {}
    for name in ('numpy', 'pandas', 'torch', 'transformers'):
        module = importlib.import_module(name)
        result[name] = dict(version=module.__version__, file=str(Path(module.__file__).resolve()))
    if result['numpy']['version'] != '1.26.4':
        raise ValueError('The pinned overlay was validated for existing NumPy 1.26.4 only')
    if sys.version_info[:2] != (3, 10):
        raise ValueError('Calibration overlay requires the actual Python 3.10 runtime')
    if importlib.import_module('torch').cuda.is_initialized():
        raise ValueError('CPU capability verification unexpectedly initialized CUDA')
    return result


def capabilities(base, overlay):
    """Run in the actual runtime child; only tiny synthetic CPU arrays are used."""
    core = core_identity()
    if any(Path(v['file']).is_relative_to(overlay) for v in core.values()):
        raise ValueError('CPU overlay must not override an existing core package')
    versions = {}
    for package, expected in PINS.items():
        module = importlib.import_module(MODULES[package])
        distribution = importlib.metadata.distribution(package)
        if (distribution.version != expected or not Path(module.__file__).resolve().is_relative_to(overlay)
                or not Path(distribution.locate_file('')).resolve().is_relative_to(overlay)):
            raise ValueError('Pinned calibration dependency is missing or resolved outside the own overlay: '+package)
        versions[package] = expected
    repo = Path(os.environ.get('QPROBE_SOURCE', str(base/'repo'))).resolve()
    if not repo.is_relative_to(base/'source'):
        raise ValueError('Capability imports must bind this namespace immutable source')
    imports = import_requests(repo)
    for name, symbol in imports:
        module = importlib.import_module(name)
        if symbol is not None and not hasattr(module, symbol):
            raise ValueError('Missing actual calibration symbol: '+name+'.'+symbol)
    # Exercise the production grouped-CV estimator code, rather than a duplicate fit.
    probe = importlib.import_module('ts_probe_train')
    import numpy as np
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    import warnings
    rng = np.random.RandomState(187)
    groups = np.repeat(np.arange(40), 4)
    y = np.tile(np.array([0, 1, 0, 1]), 40)
    x = rng.normal(size=(160, 6)); x[:, 0] += 3*y
    scores = {}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for model in ('lr', 'mlp64', 'mlp128', 'mlp256'):
            auc, folds = probe.cv_auc(x, y, groups, model, require_all_folds=True)
            if len(folds) != 2 or not np.isfinite(auc) or not 0 <= auc <= 1:
                raise ValueError('Actual calibration grouped CV failed: '+model)
            fitted = probe.make_clf(model).fit(x, y)
            before = fitted.predict_proba(x)
            with tempfile.TemporaryDirectory(prefix='overlay-cpu-smoke-', dir=base/'tmp') as temporary:
                file = Path(temporary)/'head.pkl'
                joblib.dump({'clf': fitted, 'auc': auc}, file)
                restored = joblib.load(file)
                if list(restored['clf'].classes_) != [0, 1] or not np.array_equal(before, restored['clf'].predict_proba(x)):
                    raise ValueError('Calibration joblib serialization/readback differs')
            scores[model] = dict(folds=folds, auc=auc, roundtrip_exact=True)
    if importlib.import_module('torch').cuda.is_initialized():
        raise ValueError('Calibration CPU smoke unexpectedly initialized CUDA')
    return dict(core=core, packages=versions, actual_imports=[list(row) for row in imports], cv=scores,
                source_files_sha256={n: hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in CALIBRATION},
                cpu_only=True, gpu_execution_tested=False)


def runtime_probe(base, overlay, *, baseline=False):
    module = 'reproduction.hpc3_dynrpo_periodic.environment_overlay'
    expression = ('core_identity()' if baseline else 'capabilities(Path('+repr(str(base))+'),Path('+repr(str(overlay))+'))')
    script = 'from pathlib import Path; import json; from '+module+' import core_identity,capabilities; print(json.dumps('+expression+',allow_nan=False))'
    env = clean_env(base, overlay)
    env['CUDA_VISIBLE_DEVICES'] = ''
    result = subprocess.run([str(base/'env/bin/python'), '-c', script], env=env,
                            text=True, capture_output=True, timeout=240, check=False)
    if result.returncode:
        raise RuntimeError('Actual runtime CPU capability check failed:\n'+result.stderr[-8000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def receipt_path(base):
    return base/'env/overlay-installed.json'


def check(base):
    receipt = receipt_path(base)
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError('No accepted independent calibration overlay; install it before submission')
    saved = json.loads(receipt.read_text())
    overlay = base/'env/overlay'
    if saved['base'] != str(base) or saved['pins'] != PINS or inventory(overlay) != saved['files_sha256']:
        raise ValueError('Accepted package overlay content changed')
    proof = runtime_probe(base, overlay)
    if proof['core'] != saved['core_before']:
        raise ValueError('Existing core package identity changed after the overlay installation')
    return dict(status='CALIBRATION_CPU_OVERLAY_CHECK_PASS', base=str(base), proof=proof, checked_unix=time.time())


def install(base):
    with installation_locks(base):
        queue = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T|%Z'], text=True, timeout=30)
        assert_inactive(base, queue)
        overlay = base/'env/overlay'
        if receipt_path(base).exists():
            return check(base)
        if overlay.exists():
            raise ValueError('An unaccepted partial overlay exists; retain it and reconcile before retry')
        before = runtime_probe(base, overlay, baseline=True)
        env = clean_env(base)
        pip = subprocess.run([CONTROL, '-I', '-m', 'pip', '--version'], env=env, text=True,
                             capture_output=True, timeout=30, check=False)
        if pip.returncode:
            raise RuntimeError('The control Python has no working pip; shared runtime will not be modified')
        overlay.mkdir(mode=0o770)
        result = subprocess.run(pip_command(overlay), env=env, text=True, capture_output=True, timeout=1200, check=False)
        if result.returncode:
            raise RuntimeError('Internal wheel installation failed; own partial overlay retained:\n'+result.stderr[-8000:])
        files = inventory(overlay)
        proof = runtime_probe(base, overlay)
        if proof['core'] != before:
            raise ValueError('Overlay changed an existing core package identity')
        # Re-query after pip: a concurrently submitted job must not consume an unaccepted overlay.
        assert_inactive(base, subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T|%Z'], text=True, timeout=30))
        value = dict(status='CALIBRATION_CPU_OVERLAY_INSTALLED_VERIFIED', base=str(base), pins=PINS,
                     files_sha256=files, core_before=before, proof=proof, index=INDEX,
                     shared_packages_modified=False, checked_unix=time.time())
        with receipt_path(base).open('x') as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        return dict(status=value['status'], base=str(base), packages=PINS, proof=proof, checked_unix=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=os.environ.get('SEARCH_G1_BASE', ROOT))
    parser.add_argument('--check', action='store_true', help='Read-only actual runtime import, grouped CV and joblib roundtrip verification')
    args = parser.parse_args()
    base = owned_base(args.base)
    print(json.dumps(check(base) if args.check else install(base), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
