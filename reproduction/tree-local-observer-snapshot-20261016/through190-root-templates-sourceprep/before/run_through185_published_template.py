#!/usr/bin/env python3
"""Local glue for the unchanged published through185 templates, outer185..189.

Source preparation is not execution. Run only after normal source publication.
No collector/remote operation or new validation protocol is introduced here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

W = Path(__file__).resolve().parents[2]
S = W / 'work/tree-speed-repair-20261008'
B = W / 'outputs/tree-nested-branch-handoff-20261013'
PINNED = {
    'note': ('TREE_THROUGH185_PUBLICATION_TEMPLATES_SOURCE_PREPARATION.json', 11750, 'ecd2832bfba514025dd246cf869f0f07f46b2b30a9330d473d76ae82f6543011'),
    'root_full': ('tree_through185_full_root_readback.template.command.txt', 30613, '9de64ac13275275d8f330b57d205637f3a7731ab8f49f1903f46f253bb867e64'),
    'root_verified_wait': ('tree_through185_verified_wait_readback.template.command.txt', 22688, '6d0f70fcfebe694bcce4b12b3557093caa9a8ec4530f361b345bfee8a3bc7931'),
    'finalizer': ('tree_through185_finalize.template.command.txt', 12183, '3f2c192bc73eb66d3ab512bcf9412edbf68cbcc7d263008b8c2a833286380fe5'),
}

def pin(p):
    raw = p.read_bytes()
    if not 0 < len(raw) < 4000000:
        raise ValueError('Small local metadata/template budget differs: ' + str(p))
    return raw, dict(path=str(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', required=True, choices=('finalizer', 'root_full', 'root_verified_wait'))
    parser.add_argument('--outer', required=True, type=int, choices=range(185, 190))
    parser.add_argument('--second', required=True, type=int)
    parser.add_argument('--first', type=int)
    args = parser.parse_args()
    if args.second <= 0 or args.first is not None and args.first <= 0:
        parser.error('Actual positive poll IDs are required')
    held = {}
    for key, (name, size, sha) in PINNED.items():
        raw, ref = pin(S / name)
        if ref['bytes'] != size or ref['sha256'] != sha:
            raise ValueError('Published source bytes differ: ' + name)
        held[key] = (raw, ref)
    protocol = json.loads(held['note'][0])['future_manifest_interface']
    n, first, second = args.outer, args.first, args.second
    if first is None:
        if args.role == 'root_verified_wait':
            parser.error('root_verified_wait requires its actual --first poll ID')
        full = json.loads(pin(B / f'FULL_OUTER{n}_PROGRESS_RECEIPT.json')[0])
        bound = full['input_bindings']['poll']
        match = re.fullmatch(r'NATIVE5_CONTINUE_POLL_(\d+)\.json', Path(bound['path']).name)
        if match is None:
            raise ValueError('FULL-bound original poll basename differs')
        first = int(match.group(1))
    if first == second:
        parser.error('Two distinct actual observations are required')
    role = args.role
    mirrors = [Path(x) for x in protocol['mirrors']]
    if role == 'finalizer':
        output_name = f'OUTER{n}_FULL_NEXT{n+1}_ACTUAL_PROGRESS.json'
    elif role == 'root_full':
        output_name = f'ROOT_FULL{n}_PUBLICATION_READBACK_{first}_{second}.json'
    else:
        output_name = f'ROOT_VERIFIED_WAIT_READBACK_{first}_{second}.json'
    for root in (B, *mirrors):
        if (root / output_name).exists() or role == 'root_full' and any(root.glob(f'ROOT_FULL{n}_PUBLICATION_READBACK_*.json')):
            raise ValueError('Existing successful output retained; no repeat invocation: ' + str(root / output_name))
    keys = protocol['common_exact_reference_keys'] + protocol['role_specific_exact_reference_keys'][role]
    refs = {}
    for key in keys:
        name = protocol['fixed_reference_filenames'].get(key)
        if name is None:
            name = protocol['per_boundary_reference_filenames'][key]
            name = name.replace('{n+1}', str(n+1)).format(n=n, a=first, z=second)
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Published BASE reference filename differs')
        refs[key] = pin(B / p)[1]
    manifest = dict(schema=protocol['schema'], role=role, outer=n, latest_completed_dev_outer=185,
        first_poll_index=first, second_poll_index=second, workspace_root=str(W), base=str(B),
        mirrors=[str(x) for x in mirrors], template_source=held[role][1], references=refs)
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    path = B / f'THROUGH185_{role.upper()}_ACTUAL_INPUT_PINS_OUTER{n}_{first}_{second}.json'
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Different existing manifest retained: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(raw)
    if path.read_bytes() != raw:
        raise ValueError('Actual manifest bytes changed before invocation')
    ref = pin(path)[1]
    env = os.environ.copy()
    env.update(ACTUAL_THROUGH185_PINS=str(path), ACTUAL_THROUGH185_PINS_BYTES=str(ref['bytes']),
               ACTUAL_THROUGH185_PINS_SHA256=ref['sha256'])
    command = held[role][0].decode().replace('__OUTER__', str(n)).replace('__FIRST__', str(first)).replace('__SECOND__', str(second))
    print(json.dumps(dict(actual_input_manifest=ref, published_template=held[role][1], role=role,
                          outer=n, first=first, second=second)), flush=True)
    raise SystemExit(subprocess.run(['bash'], input=command, text=True, cwd=W, env=env, check=False).returncode)


if __name__ == '__main__':
    main()
