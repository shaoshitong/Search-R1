from pathlib import Path
import json,importlib.util,hashlib,time
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-frozen-calibration-scores-v2');pub=json.loads((b/'source-publication.json').read_text());root=Path(pub['code_root']);spec=importlib.util.spec_from_file_location('frozen_score',str(root/'score_peer_g10_calibration.py'));s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);c=s.read(b/'config.json');refs=[];payloads=[]
def walk(v):
 if isinstance(v,dict):
  if 'path' in v and 'sha256' in v:
   p=Path(v['path'])
   if p.suffix in ('.pt','.safetensors'):
    assert p.is_file() and p.stat().st_size==v['bytes'];payloads.append(dict(path=str(p),bytes=v['bytes'],sha_deferred_to_allocated_GPU=True))
   else:s.checked(v);refs.append(dict(path=str(p),sha256=v['sha256'],legacy_SHA_only='bytes' not in v))
  for own in v.values():walk(own)
 elif isinstance(v,list):
  for own in v:walk(own)
walk(c);s.runtime(s.read(s.checked(c['pinned_runtime_publication'])))
for name,item in pub['files'].items():s.checked(dict(path=str(root/name),**item))
actual=s.identities(s.read(s.checked(c['groups'])));assert actual==c['identities']
r=dict(status='ACTUAL_V2_ALL_INFERENCE_METADATA_LEGACY_SHA_AND_8192_IDENTITY_PREFLIGHT_PASSED',small_metadata_refs=len(refs),legacy_SHA_only_refs=sum(r['legacy_SHA_only'] for r in refs),payloads=payloads,config=s.proof(b/'config.json'),groups=c['groups'],identities=actual,checkpoint_tensor_loaded=False,feature_payload_loaded=False,probe_updates=0,actor_updates=0,time=time.time());s.atomic(b/'actual-cpu-metadata-preflight.json',r);print(json.dumps(r))
