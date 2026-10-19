"""Start the existing5s sampler once this prepared native writer is submitted."""
import argparse
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    base = Path(args.base)
    while not (base / 'preparation/submit-receipt.json').exists():
        time.sleep(2)
    os.execv(sys.executable, [sys.executable, '-u', '-m',
        'reproduction.operations.b_replica_retained18.shared_resources', '--base', str(base)])


if __name__ == '__main__':
    main()
