"""Independent CPU acceptance and separately labeled rule replay after HPC3 v5 CF."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--job',required=True);p.add_argument('--audit-runtime',required=True)
    a=p.parse_args();job=json.loads(Path(a.job).read_text());seq=Path(job['sequence_output']);audit=Path(a.audit_runtime)
    lock=(seq/'finalize.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not(seq/'run-complete.json').is_file():raise RuntimeError('Both GPU runs/verification must complete first')
    if(seq/'final-summary.json').exists():raise RuntimeError('Already complete; never overwrite independent acceptance')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    summary=dict(state='HPC3_STRICT_V5_TWO_MODELS_VERIFIED',at=time.time(),scope='Finite-type conservative conditional subset; official changed-rule is separate same-response replay',models=[])
    for e in job['methods']:
        root=Path(e['cf_output']);accept=root/'independent-acceptance-20261019.json';replay=root/'official-changed-rule-on-v5'/'replay.json'
        commands=[]
        if not accept.exists():commands.append([job['python'],'-B',str(audit/'verify_completed_strict_v5.py'),'--manifest',str(root/'manifest.json'),'--runtime',job['runtime'],'--output',str(accept)])
        if not replay.exists():commands.append([job['python'],'-B',str(audit/'official_changed_rule_replay.py'),'--method',e['method'],'--cf-root',str(root),'--output',str(replay)])
        for n,command in enumerate(commands):
            started=time.time()
            with(seq/(e['key']+'-independent-'+str(n)+'.log')).open('ab',buffering=0)as stream:
                proc=subprocess.Popen(command,cwd=job['runtime'],env=env,stdout=stream,stderr=subprocess.STDOUT);rc=proc.wait()
            write(seq/(e['key']+'-independent-'+str(n)+'-exit.json'),dict(exit_code=rc,at=time.time(),started=started,command=command))
            if rc:raise RuntimeError('CPU acceptance/replay failed, preserve all files')
        report=json.loads(accept.read_text());assert report['state']=='INDEPENDENT_CF_FULL_READ_VERIFIED'
        s=report['cf_summary'];official=json.loads(replay.read_text())['methods'][e['method']]
        summary['models'].append(dict(method=e['method'],normal_count=s['counts']['normal_questions'],normal_correct=s['counts']['normal_correct'],counts=s['counts'],coverage=s['coverage'],TC=s['TC'],D_Inv=s['D_Inv'],TC_denominator=s['TC_denominator'],DInv_denominator=s['D_Inv_denominator'],token_usage=s['token_usage'],counterfactual_attempts=report['cf_attempts'],cf_result_bytes=report['cf_result_bytes'],cf_result_count=report['cf_results'],cf_results_hash=report['cf_results_hash'],normal_sha256=report['normal_sha256'],manifest_sha256=report['cf_manifest_sha256'],prepared_sha256=report['cf_prepared_sha256'],official_changed_rule_on_v5=official,acceptance_path=str(accept),rule_replay_path=str(replay)))
    summary['completed_at']=time.time();write(seq/'final-summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
