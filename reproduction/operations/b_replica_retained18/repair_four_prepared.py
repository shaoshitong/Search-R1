"""Reuse completed CPU reshard in a new immutable hardware-validation namespace.

No tensor load, optimizer write or repeated reshard. The original conversion
proof and original181 proof keep every target/source seal. Only placement
validation and code identity change before any native submission.
"""
import argparse,json,os,time
from pathlib import Path
from reproduction.prefix_selection200.diverse_resume import read,sha,once
from reproduction.prefix_selection200.diverse_four_resume import STATUS,UNCHANGED,DISPATCH,validate_imported

def main():
 p=argparse.ArgumentParser()
 for name in('source','base','commit'):p.add_argument('--'+name,required=True)
 p.add_argument('--outer',required=True,type=int);a=p.parse_args();original=Path(a.source);b=Path(a.base)
 prior=original.parent/'search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp181-four-shared5-20261020-r1'
 assert not(prior/'preparation/submit-intent.json').exists()
 who=read(prior/'preparation/stage-receipt.json');proc=Path('/proc')/str(who['pid'])
 if proc.exists():
  fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
  assert int(fields[19])!=who['start_ticks']or fields[0]=='Z','Do not change code of a live preparer'
 c=read(prior/'preparation/launch-config.json');j=read(prior/'latest-joint.json')
 validate_imported(prior,c)
 # Published original conversion and target files remain immutable and retained.
 migration=read(prior/'preparation/actor-cp181-8to4-reshard.json')
 for row in migration['rank_states']:
  st=Path(row['path']).stat();assert st.st_size==row['bytes']
 c.update(base=str(b),source_commit=a.commit,execution_id=__import__('uuid').uuid4().hex)
 runtime={name:sha(b/'repo/reproduction/prefix_selection200'/name)for name in UNCHANGED+DISPATCH}
 for name in UNCHANGED:assert runtime[name]==sha(original/'repo/reproduction/prefix_selection200'/name),name
 proof=dict(status=STATUS,runtime_commit=a.commit,binding_commit=c['actor_checkpoint_binding_commit'],
     source_base=str(original),source_commit=read(original/'preparation/launch-config.json')['source_commit'],
     runtime_sha256=runtime,scope='four explicit hardware/boundary dispatches, unchanged actor/probe/sampling/loss',created_unix=time.time())
 path=b/'preparation/runtime-compatibility.json';once(path,proof);c['runtime_compatibility']=dict(path=str(path),sha256=sha(path))
 once(b/'preparation/launch-config.json',c)
 def link(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
 for sub in('outers','question-pool','actor','probe'):
  for f in(prior/sub).rglob('*'):
   if f.is_file():link(f,b/sub/f.relative_to(prior/sub))
 for name in('latest-joint.json',):link(prior/name,b/name)
 for f in(prior/'preparation').glob('prefix-migration-*.json'):link(f,b/'preparation'/f.name)
 for name in('resume-joint-receipt.json','actor-cp181-8to4-reshard.json'):link(prior/'preparation'/name,b/'preparation'/name)
 from reproduction.prefix_selection200.launch import validate_config
 validate_config(c,b);validate_imported(b,c)
 ready=dict(status='LATEST_NATIVE_JOINT_PREPARED_AWAITING_TARGET',outer=181,next_outer=182,base=str(b),world_size=4,
     global_micro=4,local_micro=1,accumulation=32,prefix_tokens=80,actor=[3620]*4,probe=[2530]*4,
     optimizer_reset=False,training_started=False,source_commit=a.commit,ready_unix=time.time(),
     reused_exact_cpu_conversion=str(prior),new_optimizer_steps=0,new_tensor_loads=0)
 once(b/'preparation/native-recovery-complete.json',ready);print(json.dumps(ready),flush=True)

if __name__=='__main__':main()
