"""Pin latest durable owned HF policy and existing candidate inputs with hardlinks.

No tensor data are read/copied, trained, chmod'ed or sent to the local machine.
The ready rolling manifest supplies hashes; equal inodes preserve exact bytes
when the source later retires an old nonmilestone checkpoint.
"""
import json
import os
from pathlib import Path
import sys
import time


def main():
    request = json.load(sys.stdin)
    production = Path(request['source_base'])
    root = Path(request['root'])
    root.mkdir(parents=True, exist_ok=True)
    receipt = root/'policy-pin.json'
    if receipt.exists():
        print(receipt.read_text(), flush=True)
        return
    pointer = json.loads((production/'preparation/rolling-checkpoints/latest.json').read_text())
    if pointer['status'] != 'EASY2560_ROLLING_CHECKPOINT_READY':
        raise RuntimeError('Latest durable policy is not ready')
    manifest = json.loads(Path(pointer['manifest_path']).read_text())
    checkpoint = Path(pointer['checkpoint'])
    destination = root/('policy-step-'+str(pointer['step']))
    destination.mkdir(exist_ok=False)
    files = {}
    for name, record in manifest['files'].items():
        if '/' in name or not (name.endswith('.json') or name.endswith('.safetensors') or name.endswith('.txt')):
            continue
        source, target = checkpoint/name, destination/name
        os.link(source, target)
        a, b = source.stat(), target.stat()
        if (a.st_dev, a.st_ino, a.st_size) != (b.st_dev, b.st_ino, record['bytes']):
            raise RuntimeError('Pinned policy link differs from sealed source')
        files[name] = dict(path=str(target), bytes=b.st_size, sha256=record['sha256'],
                           device=b.st_dev, inode=b.st_ino)
    if not any(name.endswith('.safetensors') for name in files):
        raise RuntimeError('No HF model shards found')
    inputs = root/'inputs'; inputs.mkdir()
    candidate_files = []
    for source_value in request['candidate_files']:
        source = Path(source_value)
        target = inputs/source.name
        os.link(source, target)
        stat = target.stat()
        candidate_files.append(dict(path=str(target), original=str(source), bytes=stat.st_size,
                                    device=stat.st_dev, inode=stat.st_ino))
    value = dict(schema='owned_latest_hf_candidate_pair_pin_v1', checked_unix=time.time(),
        root=str(root), checkpoint=str(destination), source=pointer, files=files,
        candidate_files=candidate_files, logical_hf_bytes=sum(x['bytes'] for x in files.values()),
        logical_input_bytes=sum(x['bytes'] for x in candidate_files), new_tensor_bytes_copied=0,
        tensors_read=False, source_modified=False, source_permissions_changed=False,
        source_hash_semantics='borrowed from sealed full manifest; equal device/inode hardlinks preserve bytes',
        benchmark_optimizer_loaded=False)
    with receipt.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(value),flush=True)


if __name__ == '__main__':
    main()
