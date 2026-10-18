"""Immutable pipeline entry with explicit failure and exit metadata."""
import os
os.environ['SEARCH_G1_GRPO_FROM50']='1'
os.environ.setdefault('SEARCH_G1_GRPO_PREFIX','20')
import argparse
from pathlib import Path
import time
import traceback

from .bootstrap import read,publish


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args();base=Path(a.base)
    success=False
    try:
        from reproduction.prefix_selection200.launch import run
        run(base);success=True
    except BaseException as error:
        publish(base/'logs/failure.json',dict(checked_unix=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:publish(base/'logs/exit.json',dict(success=success,checked_unix=time.time()))


if __name__=='__main__':main()
