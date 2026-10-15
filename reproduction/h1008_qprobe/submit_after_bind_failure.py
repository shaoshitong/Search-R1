"""Isolated once-only fresh launch after a proven pre-training bind failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from reproduction.h1008_qprobe.submit import submit, once
from reproduction.h1008_qprobe.adopt_reward_zero import digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve();prep=base/'preparation'
    old=Path('/mnt/data/shaoshitong/search-g1-qprobe3-h1008-77579a4-20261015-v3')
    if (prep/'bind-recovery-intent.json').exists() or (prep/'submit-intent.json').exists():
        raise RuntimeError('Prior intent exists; inspect actual receipt, never replay')
    queue=subprocess.check_output(['squeue','-h','-w','h100-8','-o','%i|%j|%T'],text=True)
    assert not any(row.split('|')[0]=='3396' or 'search-g1-' in row for row in queue.splitlines())
    error=(old/'logs/native-3396.log').read_text()
    assert error.rstrip().endswith('ValueError: This rollout phase requires the four independent TP1 engines')
    assert not (old/'logs/actual-updates.jsonl').exists()
    assert not (old/'logs/training-launch-claim.json').exists()
    assert json.loads((old/'logs/sequence-precompute-exit.json').read_text())['exit_code']==0
    ledger=json.loads((old/'logs/sequence-precompute-process-ledger.json').read_text())
    count=0
    for job in ledger['jobs'].values():
        for row in job['ledger'].values():
            count+=1;p=Path('/proc')/str(row['pid'])/'stat'
            assert not p.exists() or p.read_text().rsplit(')',1)[1].split()[19]!=str(row['start_ticks'])
    assert count>0 and not list(Path('/sys/fs/cgroup').glob('**/job_3396'))
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert not gpu.strip()
    cache=old/'question-features'
    cache_sha=digest(cache/'manifest.json')
    assert cache_sha=='b59bc36032d54cd6359f644b9692875c31fbca73f129472e4d4dbe17e1c1235e'
    config_path=prep/'h1008-launch-config.json';config=json.loads(config_path.read_text())
    assert config['base']==str(base) and config['restore_receipt'] is None
    assert not (base/'logs/run-manifest.json').exists()
    # Only frozen official input features are reused, never actor/head/optimizer state.
    once(prep/'bind-recovery-intent.json',dict(failed_job_id=3396,failed_native_log_sha256=digest(old/'logs/native-3396.log'),
        original_process_ledger_sha256=digest(old/'logs/sequence-precompute-process-ledger.json'),original_birth_count=count,
        original_births_live=[],gpu_processes=gpu,cgroup_exists=False,prior_actor_updates=0,
        native_exit_code_unknown=True,cache=str(cache),cache_manifest_sha256=cache_sha,checked_unix=time.time()))
    (base/'question-features').symlink_to(cache,target_is_directory=True)
    config['precomputed_question_features']=dict(path=str(cache),manifest_sha256=cache_sha)
    temporary=prep/'h1008-launch-config.reuse.partial'
    once(temporary,config);os.replace(temporary,config_path)
    for name in ('predecessor-exit0.json','predecessor-teardown-closure.json'):
        with (prep/name).open('xb') as f:f.write((old/'preparation'/name).read_bytes());f.flush();os.fsync(f.fileno())
    command=[str(base/'env/bin/python'),'-m','unittest','search_g1.test_rollout_phase_contract','reproduction.h1008_qprobe.test_launch_contract','reproduction.h1008_qprobe.test_budget_profile','-v']
    test=subprocess.run(command,cwd=base/'repo',capture_output=True,text=True,timeout=120)
    once(prep/'bind-recovery-cpu-contract-tests.json',dict(command=command,returncode=test.returncode,stdout=test.stdout,stderr=test.stderr,extra_gpu_test=False))
    assert test.returncode==0,test.stderr
    receipt=submit(base,json.loads((prep/'source-files.json').read_text()))
    once(prep/'bind-recovery-receipt.json',dict(submission=receipt,official_fresh_actor=True,cache_only_reused=True,checked_unix=time.time()))
    print(json.dumps(receipt))


if __name__=='__main__':main()
