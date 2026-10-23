"""Independent two-GPU service using the measured four-shard retrieval code."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import uuid

from .infra_profile import BASE, WIKI, RETRIEVAL_PORT, own_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--port', type=int, default=RETRIEVAL_PORT)
    args = parser.parse_args()
    base = own_base(args.base)
    identity = base / 'preparation/retrieval-run-id'
    if identity.exists():
        run_id = identity.read_text().strip()
    else:
        run_id = uuid.uuid4().hex
        with identity.open('x') as f:
            f.write(run_id + '\n')
    libraries = list((base / 'env/lib/python3.10/site-packages/nvidia').glob('*/lib'))
    libraries += [base / 'env/lib/python3.10/site-packages/torch/lib']
    os.environ['LD_LIBRARY_PATH'] = ':'.join(map(str, libraries))
    sys.argv = ['retrieval_service', '--base', str(base / 'retrieval'),
        '--run-id', run_id, '--source-commit', args.commit,
        '--index', str(Path(WIKI) / 'e5_Flat.index'),
        '--corpus', str(Path(WIKI) / 'wiki-18.jsonl'),
        '--encoder', str(base / 'assets/e5-base-v2'),
        '--stage-parquet', str(base / 'data/stage.parquet'),
        '--port', str(args.port), '--host', '0.0.0.0',
        '--faiss-batch-size', '16', '--faiss-max-wait-ms', '2']
    # Do not pass training_run_id before the driver creates its run manifest;
    # the independent allocation's signed receipt binds the actual consumer.
    runpy.run_module('reproduction.easy2560_joint4.retrieval_service', run_name='__main__')


if __name__ == '__main__':
    main()
