import pathlib,json,subprocess,time,fcntl
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1')
lock=(b/'root-submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
root=b/'confirmation2048-B';p=root/'root-submission-intent.json'
assert not (root/'root-submission.json').exists()
intent=json.loads(p.read_bytes());assert abs(intent['time']-1792678451.187637)<0.01
for cmd in (['squeue','-h','-u','mou176','-n','q7b-final-cB','-o','%i'],['sacct','-X','-n','-P','-u','mou176','--name=q7b-final-cB','--starttime=2026-10-22T00:00:00','--format=JobIDRaw']):
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert r.returncode==0 and not r.stdout.strip()
rejection=dict(time=time.time(),original_intent=intent,reason='Explicit nonzero sbatch result: default-account lookup API timeout, submission failed access/permission denied; fresh squeue and sacct both empty',automatic_unknown_outcome_retry=False,next_submit_explicit_account='mou176')
(root/'root-rejection-default-account.json').write_text(json.dumps(rejection,sort_keys=True)+'\n')
p.rename(root/'root-submission-intent-rejected-default-account.json')
print(json.dumps(dict(confirmed_rejected_not_submitted='q7b-final-cB',existing_cA_not_changed=True)))
