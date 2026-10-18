"""Code-only deployment and single guarded submission for the explicit CP50 progressive continuation.

Run with python3 from the isolated local checkout. Credentials remain in the
existing RAM/PTY control adapter; no model, checkpoint or environment is copied.
"""
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from reproduction.hpc3_prefix5.control import OUTPUT, PYTHON, BASE
from reproduction.hpc3_prefix5.direct_control import run

SOURCE = Path(BASE + '-r2')
TARGET = Path(BASE + '-r3')


def remote(code, timeout=25):
    address = subprocess.check_output(['ipconfig', 'getifaddr', 'en0'], text=True).strip()
    result = run(PYTHON + ' -c ' + shlex.quote(code), account='mou176',
                 source_address=address, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout.decode()[-4000:] + result.stderr.decode()[-1000:])
    return result.stdout.decode()


commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
published = subprocess.check_output(['git', 'ls-remote', 'origin',
    'refs/heads/codex/prefix20-40-80-cp50-20261018'], text=True).split()[0]
assert commit == published
assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
files = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
files = [name for name in files if name]
publication = dict(commit=commit, remote_sha=published,
    files={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},
    checked_unix=time.time(), source_only=True)
changed = subprocess.check_output(['git', 'diff', '--name-only',
    '0e516bb33e2f6bc57f59a4694baad629b0bea947', commit], text=True).splitlines()
original_names = remote('import pathlib,json\np=pathlib.Path(%r)\nprint(json.dumps(list(json.loads(p.read_text())["files"])))\n'
                        % str(SOURCE / 'preparation/source-publication.json'))
original_names = next(json.loads(line) for line in original_names.splitlines()
                      if line.strip().startswith('['))
# The earlier code-only publisher omitted Git dotfiles. Include such unchanged
# files as well, so the complete new publication has no missing metadata.
changed = sorted(set(changed) | (set(files) - set(original_names)))
stream = io.BytesIO()
with tarfile.open(fileobj=stream, mode='w:gz') as archive:
    for name in changed:
        raw = Path(name).read_bytes()
        item = tarfile.TarInfo(name)
        item.size = len(raw)
        item.mode = Path(name).stat().st_mode & 0o777
        archive.addfile(item, io.BytesIO(raw))
archive = stream.getvalue()
remote("""import pathlib,shutil,os,json
s=pathlib.Path(%r);b=pathlib.Path(%r)
assert s.resolve()==s and s.stat().st_uid==os.getuid()
assert not (b/'preparation/submit-intent.json').exists()
assert not (b/'preparation/slurm-submit-intent.json').exists()
b.mkdir(exist_ok=True);assert b.resolve()==b and b.stat().st_uid==os.getuid()
for name in ['preparation','logs','env/bin','env/lib/python3.10','tmp','cache']:
 (b/name).mkdir(parents=True,exist_ok=True)
if not (b/'repo').exists():shutil.copytree((s/'repo').resolve(),b/'repo')
assert not (b/'repo').is_symlink()
print(json.dumps(dict(status='ISOLATED_SOURCE_CLONED',target=str(b))))
""" % (str(SOURCE), str(TARGET)))


def transfer(data, name):
    encoded = base64.b64encode(data).decode()
    path = str(TARGET / 'preparation' / name)
    for start in range(0, len(encoded), 24000):
        chunk = encoded[start:start + 24000]
        mode = 'wb' if start == 0 else 'ab'
        remote('import pathlib,base64\np=pathlib.Path(%r)\nwith p.open(%r) as f:f.write(base64.b64decode(%r))\n'
               % (path, mode, chunk))
    return path


archive_path = transfer(archive, 'repair-code.tgz')
metadata = json.dumps(publication, sort_keys=True).encode()
metadata_path = transfer(gzip.compress(metadata), 'publication.json.gz')
print(remote("""import pathlib,tarfile,hashlib,gzip,json,os
b=pathlib.Path(%r);s=pathlib.Path(%r);data=pathlib.Path(%r).read_bytes()
assert hashlib.sha256(data).hexdigest()==%r
with tarfile.open(%r,'r:gz') as archive:
 for item in archive.getmembers():
  p=pathlib.PurePosixPath(item.name)
  assert item.isfile() and not p.is_absolute() and '..' not in p.parts
  target=b/'repo'/p;target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(archive.extractfile(item).read());target.chmod(item.mode)
raw=gzip.decompress(pathlib.Path(%r).read_bytes());assert hashlib.sha256(raw).hexdigest()==%r
pub=json.loads(raw)
for name,digest in pub['files'].items():
 assert hashlib.sha256((b/'repo'/name).read_bytes()).hexdigest()==digest,name
(b/'preparation/source-publication.json').write_bytes(raw)
for dst,src in [(b/'env/overlay',s/'env/overlay'),(b/'retriever-gpu-lib',s/'retriever-gpu-lib'),
 (b/'env/lib/python3.10/site-packages',s/'env/lib/python3.10/site-packages')]:
 if not dst.exists():dst.symlink_to(src.resolve())
wrapper=(s/'env/bin/python').read_bytes().replace(str(s).encode(),str(b).encode())
(b/'env/bin/python').write_bytes(wrapper);(b/'env/bin/python').chmod(0o755)
print(json.dumps(dict(status='RECOVERY_SOURCE_ALL_FILES_SHA_VERIFIED',commit=pub['commit'],files=len(pub['files']))))
""" % (str(TARGET), str(SOURCE), archive_path, hashlib.sha256(archive).hexdigest(),
         archive_path, metadata_path, hashlib.sha256(metadata).hexdigest())), flush=True)
authorization=dict(mode='cp50_prefix20_40_80_three_epoch_v1',resume_outer=50,
    prefix_after_boundaries={'50':20,'100':40,'150':80},probe_epochs=3,candidate_limits=[512,5120],
    source_job='729624',source_stop='USER_EXPLICIT_STOP',actor_updates_1_to_50_reexecuted=False,
    probe_block_reset=False,probe_adam_reset=False,zero_expand_linear_and_moments=True,
    human_instruction_local_date='2026-10-18',human_instruction='从50iter resume；50/60等P10每次3epoch；50/100/150后prefix20/40/80；候选512到5120；短回答padding；block保持')
auth_path=transfer(json.dumps(authorization,ensure_ascii=False).encode(),'progressive-human-authorization.json')
script = """import subprocess,os,json,pathlib
b=pathlib.Path(%r);python=str(b/'env/bin/python')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(b/'tmp'),
 DYNRPO_PROGRESSIVE_PREFIX='cp50_prefix20_40_80_three_epoch_v1',DYNRPO_PREFIX_TOKENS='20')
subprocess.run([python,'-m','reproduction.hpc3_prefix5.progressive_runtime','--base',str(b),'--prepare'],env=env,check=True)
code="from pathlib import Path;import json;from reproduction.prefix_selection200.progressive_resume import plan;from reproduction.prefix_selection200.launch import validate_config;b=Path(%r);c=json.loads((b/'preparation/resume-config-template.json').read_text());validate_config(c,b);print(json.dumps(plan(b,c)))" % str(b)
subprocess.run([python,'-c',code],env=env,check=True)
""" % str(TARGET)
print(remote(script,timeout=55),flush=True)
if '--submit' not in __import__('sys').argv:
 print(json.dumps(dict(status='PREPARED_AND_QUALIFIED_NOT_SUBMITTED',base=str(TARGET),commit=commit)))
 raise SystemExit(0)
batch = '''#!/usr/bin/env bash
#SBATCH --job-name=prefix-schedule-cp50-thread5
#SBATCH --partition=emergency_acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
set -euo pipefail
base=TARGET_BASE
export DYNRPO_PROGRESSIVE_PREFIX=cp50_prefix20_40_80_three_epoch_v1
export DYNRPO_PREFIX_TOKENS=20
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME="$base/cache/hf"
export XDG_CACHE_HOME="$base/cache"
export TMPDIR="$base/tmp"
cd "$base/repo"
exec 9>"$base/preparation/build-run.lock"
flock -n 9
exec "$base/env/bin/python" -m reproduction.hpc3_prefix5.progressive_runtime --base "$base"
'''.replace('TARGET_BASE', shlex.quote(str(TARGET)))
submission = remote("""import pathlib,json,subprocess,time,os
b=pathlib.Path(%r);p=b/'preparation';receipt=p/'slurm-submission.json';intent=p/'slurm-submit-intent.json'
if receipt.exists():
 print(json.dumps(json.loads(receipt.read_text())));raise SystemExit(0)
if intent.exists():raise RuntimeError('Unknown prior submission; reconcile accounting, never resubmit blindly')
queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%%j|%%i|%%T'],text=True)
assert not any(line.split('|')[0] in ('prefix5-thread5','prefix5-thread5-cp9','prefix-schedule-cp50-thread5') for line in queue.splitlines())
assert (p/'resume-config-template.json').exists()
script=p/'resume-progressive-cp50.sbatch';script.write_text(%r);script.chmod(0o700)
with intent.open('x') as f:json.dump(dict(created_unix=time.time(),script=str(script),gpu_count=4),f)
result=subprocess.run(['sbatch','--parsable','--no-requeue','--output',str(b/'logs/slurm-%%j.log'),str(script)],capture_output=True,text=True,timeout=20)
assert result.returncode==0,(result.stdout,result.stderr)
job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
value=dict(job=job,base=str(b),partition='emergency_acd',total_physical_gpus=4,submitted_unix=time.time(),resume_outer=50)
with receipt.open('x') as f:json.dump(value,f)
print(json.dumps(value))
""" % (str(TARGET), batch))
value = next(json.loads(line) for line in submission.splitlines() if line.strip().startswith('{'))
(OUTPUT / 'progressive-cp50-submission.json').write_text(json.dumps(value, indent=2) + '\n')
(OUTPUT / 'progressive-cp50-publication.json').write_text(json.dumps(publication, indent=2) + '\n')
print(json.dumps(value), flush=True)
