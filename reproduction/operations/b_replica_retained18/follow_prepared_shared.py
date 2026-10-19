"""Observe the on-node capacity wait, then follow its single native writer."""
import argparse
import json
import os
import re
import time
from .control import OUT, remote

SCRIPT = r"""
import json,pathlib,time
p=pathlib.Path(BASE)
def read(f):
 for attempt in range(12):
  try:return json.loads(f.read_text())
  except FileNotFoundError:return None
  except json.JSONDecodeError:
   if attempt==11:raise
   time.sleep(0.25)
f=p/'logs/capacity-wait-samples.jsonl'
print(json.dumps(dict(observed_unix=time.time(),base=BASE,
 wait_receipt=read(p/'preparation/capacity-wait-receipt.json'),
 wait_status=read(p/'logs/capacity-wait-status.json'),wait_bytes=f.stat().st_size if f.exists()else0,
 activation=read(p/'preparation/shared-activation.json'),
 submit_receipt=read(p/'preparation/submit-receipt.json'),
 cleanup=read(p/'logs/cleanup-status.json'))))
""".replace('else0', 'else 0')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physical-base', required=True)
    args = parser.parse_args()
    if not args.physical_base.startswith('/mnt/data/shaoshitong/search-g1-'):
        raise ValueError('Observe retained18 task storage only')
    while True:
        evidence = OUT / ('prepared-follow-%d-transport.json' % time.time_ns())
        try:
            value = json.loads(remote('BASE='+repr(args.physical_base)+'\n'+SCRIPT,
                                     evidence, target='h100-10', established_controller=True))
        except RuntimeError as error:
            if not re.search(r'timed out|BOUNDED_ESTABLISHED_ROUTE_TIMEOUT|not responding|Connection closed|Connection reset|ssh: connect|kex_exchange_identification|Broken pipe',str(error),re.I):
                raise
            print(json.dumps(dict(observed_unix=time.time(),status='CURRENT_UNKNOWN',
                evidence=str(evidence),error=str(error),native_action='none',retry_seconds=1800)),flush=True)
            time.sleep(1800)
            continue
        value['evidence'] = str(evidence)
        (OUT/'CP171_QUEUE_NATIVE_LATEST.json').write_text(json.dumps(value,indent=2))
        print(json.dumps(dict(observed_unix=value['observed_unix'],wait_bytes=value['wait_bytes'],
              activated=bool(value['activation']),submitted=bool(value['submit_receipt']),
              fits=(value['wait_status']or{}).get('observed_footprint_fits'))),flush=True)
        if value['activation']:
            os.environ['B_SHARED_OBSERVE_BASE'] = args.physical_base
            from .follow_shared import main as follow_native
            follow_native()
            return
        receipt = value['wait_receipt']
        if not receipt or receipt['node'] not in ('h100-5','h100-8','h100-10'):
            raise RuntimeError('Require an identified on-node capacity waiter')
        health = 'WHO='+repr(receipt)+'\n'+r"""
import pathlib,json,time
p=pathlib.Path('/proc')/str(WHO['pid'])
try:
 st=(p/'stat').read_text().rsplit(')',1)[1].split()
 same=int(st[19])==WHO['start_ticks']and st[0]!='Z'
except OSError:same=False
print(json.dumps(dict(observed_unix=time.time(),same_birth=same)))
"""
        try:
            alive = json.loads(remote(health,OUT/('prepared-follow-health-%d.json'%time.time_ns()),
                                      target=receipt['node'],established_controller=True))
        except RuntimeError as error:
            if not re.search(r'timed out|BOUNDED_ESTABLISHED_ROUTE_TIMEOUT|not responding|Connection closed|Connection reset|ssh: connect|kex_exchange_identification|Broken pipe',str(error),re.I):
                raise
            print(json.dumps(dict(observed_unix=time.time(),status='CURRENT_UNKNOWN',
                error=str(error),native_action='none',retry_seconds=1800)),flush=True)
            time.sleep(1800)
            continue
        if not alive['same_birth']:
            raise RuntimeError('On-node capacity waiter exited; preserve evidence and diagnose')
        time.sleep(30)


if __name__ == '__main__':
    main()
