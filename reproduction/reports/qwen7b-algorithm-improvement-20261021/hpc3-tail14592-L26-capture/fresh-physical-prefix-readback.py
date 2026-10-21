"""CPU2 bounded final source SHA/row-link readback, no tensor or GPU."""
import base64,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
BODY='''import pathlib,json,gzip,hashlib,importlib.util,itertools,time,os,resource,signal
allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:2]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));signal.alarm(50)
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail14592-independent-L26-prepared-v1');q=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-joint124194-audited243-L26-v1/code-db5b5ac04173/reproduction/operations/qwen7b_warmup/replay_probe_layers.py');assert hashlib.sha256(q.read_bytes()).hexdigest()=='6a980269c8eab2b07845f8af88ea5aafb27d2a8ca07fbd803e3987719665deaa';s=importlib.util.spec_from_file_location('qualified_causal_prefix',q);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def check(ref):
 actual=r.fingerprints(ref['path']);assert all(actual[k]==ref[k]for k in('path','bytes','sha256'));return pathlib.Path(ref['path'])
start=time.time();pref=r.fingerprints(b/'preparation.json');prep=read(pref['path']);assert prep['questions']==14592 and prep['trajectories']==145920 and prep['source_raw_complete']['sha256']=='f789b74c3f3b9fc7c813fdf4f958c148a8c2741cd7e37f7e2be5917d1dfb10a0';check(prep['source_raw_complete']);sources={};seals={};seen=set();examples={};counts=[]
for ref in prep['selections']:
 sel=read(check(ref));want={x['trajectory_id']:x['qid']for x in sel['trajectories']};trace=sel['chosen_physical_prefix_bindings'];assert len(sel['prefix_files'])==1;source=sel['prefix_files'][0];count=0
 with gzip.open(check(source),'rt')as p,gzip.open(check(trace),'rt')as t:
  for pl,tl in itertools.zip_longest(p,t):
   assert pl is not None and tl is not None;row=json.loads(pl);binding=json.loads(tl);tid=row['trajectory_id'];assert tid not in seen and want.get(tid)==row['qid']==binding['qid']and binding['trajectory_id']==tid
   assert hashlib.sha256(r.canonical(row)).hexdigest()==binding['canonical_prefix_row_sha256']and row['prefix_sha256']==binding['prefix_body_sha256'];kind='old5'if binding['chosen_execution']=='original_canonical_old5'else'HPC3_new5'
   if kind not in examples:
    normalized=r.normalize_prefix(row,tid,row['qid'],True);examples[kind]=dict(trajectory_id_sha256=hashlib.sha256(tid.encode()).hexdigest(),canonical_prefix_row_sha256=binding['canonical_prefix_row_sha256'],causal_prefix_sha256=normalized['source_prefix_sha256'],question_tokens=normalized['question_length'],policy_tokens=normalized['policy_length'])
   for key,dest in(('physical_prefix',sources),('physical_seal',seals)):
    item=binding[key]
    if item['path']in dest:assert item==dest[item['path']]
    else:dest[item['path']]=item
   seen.add(tid);count+=1
 assert count==36480 and set(want)<=seen;counts.append(count);check(source);check(trace)
assert len(seen)==145920 and set(examples)==set(('old5','HPC3_new5'))
for item in list(sources.values())+list(seals.values()):check(item)
assert r.fingerprints(b/'preparation.json')==pref
out=dict(status='FIXED14592_PROJECTED_ROWS_PHYSICAL_SOURCE_FINAL_SHA_AND_BOTH_ACTUAL_CAUSAL_INPUTS_PASSED',preparation=pref,questions=14592,trajectories=145920,shard_counts=counts,source_prefix_files=len(sources),source_seal_files=len(seals),fresh_source_prefix_and_seal_SHA_after_projection=True,canonical_row_SHA_to_exact_physical_file_and_execution_verified=True,actual_old_and_new_core_input_examples=examples,original_source_files_read_once_final=True,prefix_processing_streamed_by_file=True,CPU_affinity=sorted(os.sched_getaffinity(0)),RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_seconds=time.time()-start,model_or_tensor_loaded=False,GPU_submitted=False,epoch=time.time());p=b/'pre-submit-physical-readback.json';raw=r.canonical(out)+bytes([10]);p.write_bytes(raw);print(json.dumps(dict(receipt=r.fingerprints(p),**out)))
'''
compile(BODY,'physical-readback-remote','exec');spec=importlib.util.spec_from_file_location('final_physical_readback_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(BODY.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
if r.returncode:raise RuntimeError(r.stdout[-1500:]+r.stderr[-1500:])
v=json.loads(r.stdout.splitlines()[-1]);Path(__file__).with_name('actual-physical-prefix-readback.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
