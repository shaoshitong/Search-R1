"""Prepare an explicit, authenticated actor/probe restart, never reset a learner."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil

from .common import DEFAULT_BASE, MODEL, REUSED, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True,type=Path)
    p.add_argument('--source-receipt',required=True,type=Path);p.add_argument('--commit',required=True)
    args=p.parse_args();base=args.base
    if (base.resolve()!=base or base.is_symlink() or base.stat().st_uid!=os.getuid() or
            re.fullmatch(re.escape(str(DEFAULT_BASE))+r'-r[1-9][0-9]*',str(base)) is None):
        raise ValueError('Restart requires a new independently owned numbered namespace')
    if (base/'logs/run-manifest.json').exists():
        raise ValueError('Restart already prepared; reconcile it instead of duplicating')
    from search_g1.qprobe_checkpoint import read_source,PAIR_STATUS
    _,receipt,_,origin,parent=read_source(args.source_receipt)
    if (receipt['status']!=PAIR_STATUS or receipt['step']>=200 or
            parent['question_probe']['head_kind']!='qwen_native_periodic_lastblock_binary_v1'):
        raise ValueError('Restart requires an actual complete or pending periodic actor/probe pair')
    publication=json.loads((base/'preparation/source-publication.json').read_text())
    if publication['commit']!=args.commit or not publication['pushed_and_fetched']:
        raise ValueError('Restart source must be published and verified')
    for name,digest in publication['files'].items():
        if sha(base/'source'/args.commit/name)!=digest:
            raise ValueError('Restart source changed: '+name)
    manifest=base/'preparation/official-model-manifest.json'
    shutil.copyfile(origin/'preparation/official-model-manifest.json',manifest)
    os.environ.update(SEARCH_G1_TRAINING_WORLD_SIZE='4',
        SEARCH_G1_EXECUTION_PROFILE='native4_bf16_varlen_fp32_master')
    from reproduction.common8.prepare_run import prepare
    run=prepare(base,MODEL,manifest,args.commit,rollout_memory_utilization=.1,
        local_retrieval_assets=str(REUSED/'source/assets-root/assets/wiki'),
        restore_receipt=args.source_receipt,execution_site='hpc3')
    from reproduction.common8.configuration import resolved
    resolved(base)
    print(json.dumps(dict(run_id=run['run_id'],completed_outer=receipt['step'],
        execution_phase=receipt.get('execution_phase'),actual_gpu_restore_verified=False,
        optimizer_reset=False,student_prefix_recomputed=False)))


if __name__=='__main__':main()
