"""Narrow zero-write CP50/outer51 retry preserving all native samples."""
import json
from pathlib import Path
from .diverse_resume import read,sha
BINDING_COMMIT='c3ced09047dd4ccd66e5d232f36b5952c178a7ce'

def validate_runtime_compatibility(c):
 ref=c['runtime_compatibility'];proof=read(ref['path'])
 if proof.get('status')=='DIVERSE_CP181_EIGHT_TO_FOUR_FULL_STATE':
  from .diverse_four_resume import validate_compatibility
  return validate_compatibility(c,proof)
 if proof.get('status')=='DIVERSE_SAME_EIGHT_NATIVE_RECOVERY':
  from .diverse_native_resume import validate_compatibility
  return validate_compatibility(c,proof)
 if (sha(ref['path'])!=ref['sha256'] or proof['status']!='EIGHT_MONITOR_MAPPING_ZERO_WRITE_REPAIR'
     or proof['binding_commit']!=BINDING_COMMIT or c['actor_checkpoint_binding_commit']!=BINDING_COMMIT
     or proof['runtime_commit']!=c['source_commit'] or proof['source_actual_writes']!=[0]*8
     or proof['complete_outer']!=50 or c['world_size']!=8 or c['actor_global_micro_batch_size']!=8):
  raise ValueError('Explicit immutable eight-rank monitor-only source compatibility required')
 return proof

def validate_cached_sampling(base,c):
 proof=validate_runtime_compatibility(c);spec=c['resume_cached_sampling'];root=Path(base)/'outers/outer-000051'
 source=Path(proof['source_base']);original=source/'outers/outer-000051'
 if spec['outer']!=51 or spec['source_base']!=str(source):raise ValueError('Only completed zero-write outer51 sampling')
 for relative,item in spec['seals'].items():
  if sha(original/relative)!=item or sha(root/relative)!=item:raise ValueError('Cached actual phase/selection metadata differs')
 full=read(root/'full/complete.json');scan=read(root/'prefix/complete.json');score=read(root/'probe-score/manifest.json')
 actor=read(Path(base)/'actor/latest.json');selected=root/'probe-score/actual-em-selection.json'
 if (full['run_id']!=c['run_id'] or full['outer']!=51 or full['actor_version']!=actor['actor_version']
     or full['trajectories']!=2560 or scan['actor_version']!=actor['actor_version']
     or score['probe_before_checkpoint_sha256']!=read(Path(base)/'probe/latest.json')['sha256']
     or full['actor_selection_sha256']!=sha(selected)):
  raise ValueError('Retained native sampling actor/probe/selection boundary differs')
 return selected
