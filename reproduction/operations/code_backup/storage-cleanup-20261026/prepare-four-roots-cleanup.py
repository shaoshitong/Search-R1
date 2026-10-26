import json,pathlib,os,importlib.util,collections,hashlib
A=pathlib.Path('/data/user/jxu957/storage-cleanup-audit-20261026')
R='/data/user/jxu957/experiments/'
T=R+'topd-20261008';C=R+'cosearch-continue300-x203-20261008';O=R+'dynrpo7b-formal-20261023-recoveryfix-r6';F=R+'easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022'
d=json.loads((A/'four-roots-inventory.json').read_text())
protected=[T+'/bundle/student_initial',T+'/bundle/datasets',T+'/bundle/retrieval',T+'/bundle/code',T+'/runtime/assets/common-instruct-0b002b',T+'/runtime/assets/qwen2.5-3b-instruct-aa8e725',T+'/runtime/assets/compiler-headers',T+'/runtime/assets/hpc3-compiler-headers',T+'/runtime/repo-terminal-fixed413',T+'/runtime/tools',T+'/runtime/test-tools',T+'/environment/topd-py310/lib/python3.10/site-packages/flash_attn_3_cuda.cpython-310-x86_64-linux-gnu.so',T+'/environment/topd-py310/lib/python3.10/site-packages/flash_attn_interface.py',C+'/source/env',C+'/source/retriever-gpu-lib',C+'/runtime/gpu-faiss-alignment-candidate',C+'/runtime/retriever-python310-candidate',C+'/runtime/common-score-v4-tools',C+'/runtime/cosearch-instruct200-common-em-20261011/common-training',C+'/runtime/cosearch-instruct200-common-em-20261011/common-validation',C+'/runtime/cosearch-instruct200-common-em-20261011/common-historical-validation',C+'/migration-flash-build-20261009',O+'/checkpoints/search-g1-common8/actor/global_step_50',O+'/preparation/dynrpo7b-checkpoints/global_step_50',O+'/artifacts/question-probe/update-0050',O+'/artifacts/question-probe/attempts/05b0ab9f03bf4fb2a530e21d4986f0d8',O+'/artifacts/round_50-attempt-b7035002a51d416e9616f4162d6eef74',O+'/source',O+'/repo',F+'/source']
# Preserve every library actually loaded by the current training wrapper, including transitive FAISS libraries.
imports=json.loads((A/'current-import-before.json').read_text());assert imports['returncode']==0
protected+=json.loads(imports['stdout'])['experiment_library_paths']
protected=sorted(set(os.path.realpath(x) for x in protected))
spec=importlib.util.spec_from_file_location('clean',str(A/'large-cleanup.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
files=[];retained=[];groups=collections.defaultdict(int)
for x in d['entries']:
 p=x['path']
 if not os.path.isfile(p):continue
 if any(m.within(p,k) for k in protected):retained.append(x);continue
 if m.eligible(p,x['size']):
  files.append(x);groups['/'.join(p.split('/')[:8])]+=x['blocks']
plan={'allowed_roots':d['roots'],'protected_paths':protected,'files':files,'authorization':'2026-10-26 user: preserve current dependencies; delete other large files in four named roots, retain text logs and small records. Keep original Qwen3B/7B and original datasets.'}
assert not (A/'batch9-plan.json').exists()
(A/'batch9-plan.json').write_text(json.dumps(plan));(A/'batch9-protected-before.json').write_text(json.dumps(retained))
print(json.dumps({'planned_files':len(files),'candidate_allocated_bytes_including_hardlinks':sum(x['blocks'] for x in files),'protected_files':len(retained),'groups':sorted(groups.items(),key=lambda x:-x[1]),'code_sha256':hashlib.sha256((A/'large-cleanup.py').read_bytes()).hexdigest()}))
