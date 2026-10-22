import sys,json,hashlib,shutil,time
from pathlib import Path
commit='c7976da3afc863a1c20b99daa7cc99de3bb61f15'
authorization=json.loads('{"user_message_id": "01a163d2-2b6c-7913-b548-c71fc308bbd3", "original_utc": "2026-10-22T13:34:40.492000+00:00", "timestamp_source": "UUIDv7 user message identifier from current Codex thread", "thread_id": "01a163ab-97c3-7831-9d1d-41be25280063", "user_text": "\\u5bf9\\u7684\\uff0c\\u6838\\u5fc3\\u5c31\\u662f\\u4f60\\u6d4b\\u8bd5\\u5b8c\\u4e86\\u4e4b\\u540e\\u901f\\u5ea6\\u53d8\\u5feb\\u4e86,\\u7136\\u540e\\u90a3\\u4e48\\u6211\\u4eec\\u5c31\\u53ef\\u4ee5\\u5c31\\u662f\\u66ff\\u6362\\u5230\\u539f\\u6765\\u7684\\u4e00\\u4e2a\\u8bad\\u7ec3\\u3002", "recorded_unix": 1792684704.773242, "scope": "Apply the existing conditional faster-equivalent-infrastructure continuation authorization to the same ongoing run, currently own continuation749255, at its next complete full checkpoint. Preserve all native4 model/Adam/scheduler/RNG/data state and total200. No hot edits.", "interpretation": "The named current source is a continuation of the original authorized training run; this is the validated decoder optimization within the same performance goal, not new algorithm work.", "source_job": "749255", "source_base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-lean-r1-20261022", "source_commit": "48b87a7399a25f78b4fcf077e946a5a2b75ecace", "source_run_id": "232f20e112ac4a978c5aba52a0367195", "target_base": "/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r1-20261022", "target_runtime_commit": "c7976da3afc863a1c20b99daa7cc99de3bb61f15", "minimum_completed_outer": 54, "qualification_status": "ACTUAL_CPU17_787X_AND_ALL37907_CUTS_IDENTICAL_48_TESTS_PASS", "native_prefix_cpu_speedup": 17.787318236156125, "full_saved_actions": 37907, "full_cut_mismatches": 0, "model_forward_loss_reward_sampling_unmodified": true, "current_boundary_complete_not_yet_observed": true, "production_stop_allowed_now": false, "stop_condition": "Only after actual latest full checkpoint>=54 exists for749255/48b; preserve artifacts and use exact owned Slurm job identity. Final200/no201 unchanged."}')
base=Path(authorization['target_base']);source=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source')/commit
target=base/'source'/commit
if not target.exists():
 stage=target.with_name('.partial-'+commit);stage.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,stage);stage.rename(target)
proof=json.loads((target/'source-publication.json').read_text());assert proof['source_commit']==commit
for name,v in proof['files'].items():
 raw=(target/name).read_bytes();assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
sys.path.insert(0,str(target))
from reproduction.easy2560_joint4.control import publish
p=base/'preparation/native-prefix-transition-authorization.json'
if p.exists():assert json.loads(p.read_text())==authorization
else:publish(p,authorization)
from reproduction.operations.easy7b_20261021.fa3_lean_transition import source_binding
with source_binding(source_base=authorization['source_base'],source_job=authorization['source_job'],source_commit=authorization['source_commit'],authorization=authorization,minimum_outer=54) as bound:
 result=bound.prepare(str(base),commit)
print(json.dumps({'epoch':time.time(),'result':result,'source_verified_files':len(proof['files']),'training_submitted':False,'source_stopped':False}))
