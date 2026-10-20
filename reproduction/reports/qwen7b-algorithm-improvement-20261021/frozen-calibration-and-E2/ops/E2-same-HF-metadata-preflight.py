from pathlib import Path
import importlib.util,json,time
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/E2-same-HF-baseline-20261021');pub=json.loads((b/'source-publication.json').read_text());root=Path(pub['code_root']);spec=importlib.util.spec_from_file_location('score',str(root/'score_peer_g10_calibration.py'));s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);c=s.read(b/'config.json');refs=[];payloads=[]
def walk(v):
 if isinstance(v,dict):
  if 'path' in v and 'sha256' in v:
   p=Path(v['path'])
   if p.suffix=='.pt':assert p.stat().st_size==v['bytes'];payloads.append(dict(path=str(p),bytes=v['bytes'],full_SHA_inside_GPU=True))
   else:s.checked(v);refs.append(str(p))
  for x in v.values():walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
walk(c);s.runtime(s.read(c['pinned_runtime_publication']['path']))
for name,r in pub['files'].items():s.checked(dict(path=str(root/name),**r))
g=s.read(s.checked(c['groups']));assert len(g)==2048 and len({x['qid'] for x in g})==2048 and len({t for x in g for t in x['trajectory_ids']})==10240 and all(len(x['trajectory_ids'])==len(x['labels'])==5 for x in g)
r=dict(status='E2_ACTUAL_ALL_SMALL_INFERENCE_METADATA_AND_FROZEN_G5_IDS_PASSED',config=s.proof(b/'config.json'),metadata_refs=len(refs),payloads=payloads,questions=2048,trajectories=10240,no_checkpoint_tensor_loaded=True,no_feature_payload_loaded=True,time=time.time());s.atomic(b/'actual-cpu-metadata-preflight.json',r);print(json.dumps(r))
