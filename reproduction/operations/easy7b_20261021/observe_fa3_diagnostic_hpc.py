"""Read one existing isolated FA3 diagnostic claim; never submit or mutate it."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    request = json.load(sys.stdin)
    mode = request.get('mode', 'model')
    if mode not in ('model', 'layer-shadow'):
        raise ValueError('Unknown diagnostic mode')
    suffix = 'model-probe' if mode == 'model' else 'layer-shadow'
    base = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-'+suffix+'-20261021')
    job = (base/'job.id').read_text().strip()
    if not job.isdecimal():
        raise ValueError('Existing numeric diagnostic job required')
    result = dict(epoch=time.time(), read_only=True, job=job, base=str(base), mode=mode)
    commands = {
        'queue':['squeue','-j',job+',745371,745198','-h','-o','%i|%T|%N|%R|%S|%L'],
        'accounting':['sacct','-j',job,'-n','-P','-o','JobIDRaw,State,ExitCode,Elapsed,NodeList'],
    }
    for name, command in commands.items():
        process = subprocess.run(command, capture_output=True, text=True, timeout=12)
        result[name] = dict(rc=process.returncode, stdout=process.stdout, stderr=process.stderr)
    report = 'diagnostics/'+('fa3-model-comparison.json' if mode == 'model' else 'fa3-layer-shadow.json')
    members = ['logs/model-probe-source-start.json','logs/model-probe-prepared.json', report]
    members += [str(path.relative_to(base)) for path in (base/'diagnostics').glob('*failure.json')]
    for name in members:
        path = base/name
        if path.is_file():
            if path.stat().st_size > 4*1024**2:
                raise ValueError('Diagnostic JSON unexpectedly large')
            result[name] = json.loads(path.read_text())
    log = base/('logs/model-probe-'+job+'.log')
    if log.is_file():
        stat = log.stat()
        with log.open('rb') as stream:
            stream.seek(max(0,stat.st_size-12000))
            result['log'] = dict(bytes=stat.st_size,mtime=stat.st_mtime,tail=stream.read().decode(errors='replace'))
    receipt = base/'preparation/model-probe-source-receipt.json'
    if receipt.is_file():
        result['source_receipt'] = dict(bytes=receipt.stat().st_size,step=json.loads(receipt.read_text()).get('step'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
