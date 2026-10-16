"""Stream only the immutable epoch68 dataset archive to A800, without Mac staging.

Both physical SSH paths are the already configured project paths. No HF token,
model/checkpoint archive, foreign process, or global networking change is used.
Interrupted destination parts remain recoverable; a full SHA precedes rename.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
REFERENCE = ROOT/'qprobe-prefix20-online1000-h1008-20261016'
READY = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix20-github-export-20261016T044748Z-111a2d/UPLOAD_READY.json')
ARCHIVE = 'dataset-through-epoch-0068.tar.gz'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_command(code):
    transport = load('existing_reference_transport', REFERENCE/'reproduction/h1008_qprobe/management_transport.py')
    management = transport.MGMT.replace('WORKER_CMD', repr('/usr/bin/python3 -'))
    old = 'capture_output=True,timeout=240)'
    assert management.count(old) == 1
    management = management.replace(old, 'stdout=sys.stdout.buffer,stderr=sys.stderr.buffer,timeout=3600)')
    pos = management.index("print(json.dumps({'route':")
    management = management[:pos] + 'sys.exit(r.returncode)\n'
    args = ['env', '-u', 'LD_LIBRARY_PATH', '/usr/bin/ssh', '-T', '-o', 'BatchMode=yes',
            '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no',
            '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
            'mgmt-host', 'env -u LD_LIBRARY_PATH /usr/bin/python3 -c ' + shlex.quote(management)]
    return args, code.encode()


def target_command(code):
    module = load('existing_a800_dataset_controller', Path(__file__).with_name('a800_qprobe_prefix_scale20.py'))
    return module.route().ssh_command(code)


RECEIVE = r'''
import hashlib,json,os,pathlib,sys,time
base=pathlib.Path(BASE);root=base/'source-archives'
assert str(base).startswith('/data/shaoshitong/Search-G1-question5-ema-joint200-20261016-')
assert len(base.name.rsplit('-',1)[-1])==32 and base.resolve()==base
base.mkdir(mode=0o700,exist_ok=True);root.mkdir(mode=0o700,exist_ok=True)
assert root.resolve()==root and root.stat().st_uid==os.getuid()
entry=ENTRY;p=root/entry['name'];part=p.with_name(p.name+'.partial')
assert p.parent==root and '/' not in entry['name'] and not p.is_symlink() and not part.is_symlink()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
if p.exists():
 assert p.stat().st_size==entry['bytes'] and digest(p)==entry['sha256']
 print(json.dumps(dict(stage='ALREADY_VERIFIED',offset=entry['bytes'])),flush=True);sys.exit(0)
offset=part.stat().st_size if part.exists() else 0
assert 0<=offset<=entry['bytes']
print(json.dumps(dict(stage='READY',offset=offset)),flush=True)
with part.open('ab') as f:
 remaining=entry['bytes']-offset
 while remaining:
  raw=sys.stdin.buffer.read(min(1024**2,remaining))
  if not raw:raise EOFError('Source stream ended before complete part')
  f.write(raw);remaining-=len(raw)
 f.flush();os.fsync(f.fileno())
assert part.stat().st_size==entry['bytes'] and digest(part)==entry['sha256']
os.replace(part,p);fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
receipt=dict(stage='TRANSFER_PART_SHA_VERIFIED',checked_unix=time.time(),name=p.name,bytes=p.stat().st_size,sha256=entry['sha256'])
with p.with_name(p.name+'.verified.json').open('w') as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intent', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    args = parser.parse_args()
    intent = json.loads(args.intent.read_text())
    assert intent['prefix_tokens'] == 5 and intent['epochs'] == 200
    published = subprocess.check_output(['git', '-C', str(ROOT/'repo'), 'rev-parse', 'HEAD'], text=True).strip()
    rel = 'reproduction/operations/'+Path(__file__).name
    assert subprocess.check_output(['git', '-C', str(ROOT/'repo'), 'show', published+':'+rel]) == Path(__file__).read_bytes()
    remote = subprocess.check_output(['git', '-C', str(ROOT/'repo'), 'ls-remote', 'origin', 'refs/heads/search-g1'], text=True).split()[0]
    assert remote == published, 'Publish transfer source before execution'
    ready = json.loads(READY.read_text())
    assets = sorted((x for x in ready['assets'] if x['archive'] == ARCHIVE), key=lambda x: x['name'])
    assert len(assets) == 8 and sum(x['bytes'] for x in assets) == 8389298946
    records, copied = [], 0
    def record(**extra):
        value = dict(checked_unix=time.time(), base=intent['base'], source_commit=published,
                     total_bytes=8389298946, transferred_bytes_this_invocation=copied,
                     verified_parts=records, model_or_checkpoint_transfer=False, local_payload_staging=False, **extra)
        args.progress.write_text(json.dumps(value, indent=2)+'\n')
    record(stage='STARTING_IMMUTABLE_ARCHIVE_TRANSFER')
    for entry in assets:
        target = RECEIVE.replace('BASE', repr(intent['base'])).replace('ENTRY', repr(entry))
        with subprocess.Popen(target_command(target), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as dst:
            first = json.loads(dst.stdout.readline())
            if first['stage'] == 'ALREADY_VERIFIED':
                dst.stdin.close();dst.wait(timeout=30)
                assert dst.returncode == 0
                records.append(dict(name=entry['name'], bytes=entry['bytes'], sha256=entry['sha256'], existing_verified=True))
                record(stage='PART_ALREADY_VERIFIED');continue
            offset = first['offset']
            source = ("import pathlib,sys\np=pathlib.Path("+repr(entry['remote_path'])+")\n"
                      "assert p.stat().st_size=="+str(entry['bytes'])+"\n"
                      "with p.open('rb') as f:\n f.seek("+str(offset)+")\n"
                      " while True:\n  x=f.read(1024**2)\n  if not x:break\n  sys.stdout.buffer.write(x)\n"
                      "sys.stdout.buffer.flush()\n")
            command, code = source_command(source)
            with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as src:
                src.stdin.write(code);src.stdin.close()
                count, last = offset, time.monotonic()
                while count < entry['bytes']:
                    block = src.stdout.read(min(1024**2, entry['bytes']-count))
                    if not block:
                        raise EOFError('Source failed: '+src.stderr.read(4096).decode(errors='replace'))
                    dst.stdin.write(block);count += len(block);copied += len(block)
                    if time.monotonic()-last >= 10:
                        record(stage='STREAMING_ARCHIVE_PART', current_part=entry['name'], current_part_bytes=count)
                        last = time.monotonic()
                dst.stdin.close()
                src.wait(timeout=60)
                assert src.returncode == 0, src.stderr.read(4096).decode(errors='replace')
            result = dst.stdout.read(16384);dst.wait(timeout=300)
            assert dst.returncode == 0, dst.stderr.read(4096).decode(errors='replace')
            receipt = json.loads(result)
            assert receipt['sha256'] == entry['sha256'] and receipt['bytes'] == entry['bytes']
            records.append(receipt);record(stage='PART_SHA_VERIFIED')
    record(stage='ALL_EIGHT_DATASET_ARCHIVE_PARTS_VERIFIED', complete=True)
    print(json.dumps(dict(complete=True, progress=str(args.progress), verified_parts=len(records), base=intent['base'])))


if __name__ == '__main__':
    main()
