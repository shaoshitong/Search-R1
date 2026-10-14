"""Wait remotely for the completed evaluation, then consume one submission intent."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(path.read_text())


def readiness(base):
    config = read(base / 'preparation/h1008-launch-config.json')
    predecessor = Path(config['predecessor_evaluation_base'])
    terminal_base = Path(config.get('predecessor_terminal_evidence_base', str(base)))
    terminal_path = terminal_base / 'preparation/predecessor-exit0.json'
    if not terminal_path.exists():
        return False
    terminal = read(terminal_path)
    if terminal.get('job_id') != 3395 or terminal.get('state') != 'COMPLETED' or terminal.get('exit_code') != '0:0':
        raise RuntimeError('Predecessor native terminal is not successful; do not submit')
    paths = [predecessor / 'journal/final-result.json', predecessor / 'logs/cleanup-status.json']
    if not all(path.is_file() for path in paths):
        return False
    # The frozen submitter verifies identities, counts, source hashes and queue again.
    return True


def publish(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(base, deadline_seconds=14400):
    base = Path(base).resolve()
    status = base / 'logs/start-after-predecessor-status.json'
    sys.path.insert(0, str(base / 'repo'))
    from reproduction.h1008_qprobe.submit import submit
    started = time.monotonic()
    publish(status, dict(stage='WAITING_FOR_PREDECESSOR', pid=os.getpid(),
                         started_unix=time.time(), gpu_use=False, submitted=False))
    try:
        while time.monotonic() - started < deadline_seconds:
            # An earlier ambiguous submission must be investigated, never repeated.
            if any((base / 'preparation' / name).exists() for name in ('submit-intent.json', 'submit-receipt.json')):
                raise RuntimeError('Existing submission intent/receipt; inspect outcome, no replay')
            if readiness(base):
                config = read(base / 'preparation/h1008-launch-config.json')
                terminal_base = Path(config.get('predecessor_terminal_evidence_base', str(base)))
                target = base / 'preparation/predecessor-exit0.json'
                if terminal_base != base and not target.exists():
                    # Preserve actual terminal evidence from the original read-only
                    # observer across an unstarted candidate redeployment.
                    with target.open('xb') as handle:
                        handle.write((terminal_base / 'preparation/predecessor-exit0.json').read_bytes())
                        handle.flush()
                        os.fsync(handle.fileno())
                receipt = submit(base, read(base / 'preparation/source-files.json'))
                publish(status, dict(stage='SUBMITTED_ONCE', submitted=True,
                                     receipt=receipt, checked_unix=time.time()))
                return receipt
            time.sleep(15)
        raise TimeoutError('Predecessor readiness deadline; no submission attempted')
    except BaseException as error:
        publish(status, dict(stage='ATTENTION_REQUIRED', error=repr(error),
                             checked_unix=time.time(),
                             submission_intent_exists=(base / 'preparation/submit-intent.json').exists(),
                             submission_receipt_exists=(base / 'preparation/submit-receipt.json').exists()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.base)))
