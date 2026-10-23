"""CPU replay of the retained real outer4 probability boundary, remote only."""
from reproduction.dynrpo7b.local_control import remote

SCRIPT = r'''
import subprocess,os
src='/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r5/source/3f1650d477daa5c4f3bbae568175bf8638b9d831'
code=r"""
import json,pathlib,torch,time
from reproduction.common8.behavior_probability import verify
from reproduction.common8.behavior_trace import project_sampling_logprobs
from reproduction.common8.behavior_runtime import _bindings
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r4')
pointer=json.loads((b/'preparation/dynrpo7b-collection/outer-0004.json').read_text())
commit=torch.load(pointer['checkpoint']['path'],map_location='cpu',weights_only=False)
saved=torch.load(commit['rollout']['path'],map_location='cpu',weights_only=False,mmap=True)
out=saved['output']; width=out.batch['responses'].shape[1]
trace=json.loads((b/'artifacts/behavior-traces/attempt-e6146f768bc94649b51d6bdd98fb1d81.json').read_text())
bindings=_bindings(saved['identity'],trace['sampling']);rows=trace['actor_rows']
projected=project_sampling_logprobs(trace,actor_rows=rows,**bindings)
proof=verify(None,out.batch['sampling_log_probs'],out.batch['info_mask'][:,-width:],out.batch['attention_mask'][:,-width:],trace=trace,actor_rows=rows,audit=projected['audit'],require_hf_parity=False,**bindings)
result={k:proof[k] for k in ('accepted','logical_update','trajectory_count','included_actor_token_count','actual_frozen_old_all_finite','hf_recomputation_performed','trace_sha256')}
result.update(status='REAL_RETAINED_OUTER4_PROBABILITY_CPU_PASS',original_producers=trace['generation_producers'],GPU_used=False,checked_unix=time.time())
p=pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r5/preparation');p.mkdir(exist_ok=True,parents=True)
(p/'retained-outer4-probability-cpu.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
"""
r=subprocess.run(['/data/user/jxu957/experiments/dynrpo7b-formal-20261023/env/bin/python','-c',code],cwd=src,env=dict(os.environ,QPROBE_SOURCE=src),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120)
print(r.stdout);print(r.stderr[-2500:]);raise SystemExit(r.returncode)
'''

if __name__ == "__main__":
    remote(SCRIPT)

