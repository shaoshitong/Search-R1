"""One read-only stdlib actual-schema check; no final join or tensor loading."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

R = Path(__file__).resolve().parents[4]
CONTROL = R.parents[1]
OUT = Path(__file__).resolve().parent
PKG = 'reproduction/operations/qwen7b_warmup/'
A = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
CODE = A + '/joint124194-native-training-bank-v1/code-2c353df424fa'
FILES = {PKG + n: hashlib.sha256((R / PKG / n).read_bytes()).hexdigest()
         for n in ('joint124194_bank.py', 'replay_probe_layers.py',
                   'replay_feature_reader.py', 'g10_grouping.py')}

SCRIPT = '''import hashlib,json,os,pathlib,resource,sys,time
t0=time.monotonic()
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:2]))
os.environ['CUDA_VISIBLE_DEVICES']=''
code=pathlib.Path(v['code'])
for name,expected in v['files'].items():
 assert hashlib.sha256((code/name).read_bytes()).hexdigest()==expected, name
sys.path.insert(0,str(code))
from reproduction.operations.qwen7b_warmup import joint124194_bank as j
r=j.r
def proof(path):return r.fingerprints(path)
def readref(ref):return r.read(r.checked_file(ref)['path'])
publication=proof(code.parent/'source-publication.json')
pub=r.read(publication['path']);assert pub['commit'].startswith('2c353df424fa')
existing=proof(v['A']+'/peer-threebank58432-internal5120-plan-v1/complete.json')
recovered=proof(v['A']+'/peer-joint124194-recovered1216-L26-v1/bank/complete.json')
assert existing['sha256']=='8a4ed0b24296ad8a5edc1bcecb249565b877d3bababb3f0221a8e7c26da72d06'
assert recovered['sha256']=='cedfa2e1892b03803182ec38a3d1c8d159167a62294e646858c3563d0942672a'
old=r.read(existing['path']);new=r.read(recovered['path'])
assert old['status']=='THREE_ACCEPTED_BANKS_58432_INTERNAL5120_FIT53312_APPEND728_METADATA_ACCEPTED'
assert new['status']==j.SUBSET_STATUS
assert (old['questions'],old['trajectories'],new['questions'],new['trajectories'])==(58432,584320,1216,12160)
assert not old['actor_updates'] and not new['actor_updates']
lineage=j.subset_lineage(dict(key='recovered1216',questions=1216,complete=recovered,
 expected_complete_sha256=recovered['sha256']),new)
print(json.dumps(dict(stage='ACTUAL_RECOVERED1216_SUBSET_LINEAGE_ACCEPTED',elapsed_seconds=time.monotonic()-t0)),flush=True)
exref=dict(path=v['A']+'/joint-balanced-native-E4-qualification-v3/evaluation-identity-exclusions.json',bytes=847206,
 sha256='a4ed20a3732984d13a099494d7bbb599aeb05b2868945b4f1f450786963b9b0a')
ex=readref(exref);assert ex['role']=='joint_training_fixed_evaluation_identity_exclusions'
assert len(ex['groups'])==6656 and len(set(g['qid']for g in ex['groups']))==6656
oldgroups=readref(old['outputs']['merged-bank/groups.json'])['train']
newgroups=readref(new['outputs']['groups.json'])['train']
merged,summary=j.merge_groups(dict(existing58432=oldgroups,recovered1216=newgroups),ex['groups'],
 expected_counts=dict(existing58432=58432,recovered1216=1216),
 expected_domains=dict(nq=43660,hotpotqa=15988),expected_text_repeats=0)
assert len(merged)==59648
print(json.dumps(dict(stage='ACTUAL_TWO_BANK_59648_OWN_ID_EM_MERGE_ACCEPTED',questions=summary['questions'],
 trajectories=summary['trajectories'],elapsed_seconds=time.monotonic()-t0)),flush=True)
refs=readref(new['outputs']['features-layer26.json']);assert len(refs)==40
accepted,features=j.verify_features(refs,newgroups)
assert features['manifests']==40 and features['own_trajectory_rows']==12160
assert not features['payload_tensor_SHA_rehashed_in_this_metadata_join']
assert 'torch'not in sys.modules and 'numpy'not in sys.modules
print(json.dumps(dict(status='ACTUAL_TWO_BANK_SCHEMA_LINEAGE_AND_RECOVERED1216_INDEX_CHECK_PASSED',
 source_publication=publication,source_commit=pub['commit'],checked_code_SHA256=v['files'],
 source_completions=dict(existing58432=existing,recovered1216=recovered),
 true_evaluation_identity_exclusions=exref,exclusion_questions=6656,
 subset_lineage=lineage,merged_summary=summary,recovered1216_features=features,
 elapsed_seconds=time.monotonic()-t0,max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 CPU_affinity=sorted(os.sched_getaffinity(0)),GPU_used=False,model_or_tensor_loaded=False,
 remote_artifacts_written=False,final124194_join_written=False,external_or_final_outcomes_read=False,
 epoch=time.time()),sort_keys=True),flush=True)
'''

def main():
    code = 'v=' + repr(dict(code=CODE, files=FILES, A=A)) + '\n' + SCRIPT
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', prefix='q7b-partial-schema-', delete=False) as f:
        f.write(code)
        script = Path(f.name)
    try:
        result = subprocess.run(['python3', str(CONTROL/'outputs/qwen7b-warmup-20261019/balanced-control.py'),
                                 str(script), '--timeout', '150'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=165)
    finally:
        script.unlink()
    text = result.stdout.decode(errors='replace')
    rows = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    OUT.mkdir(exist_ok=True)
    (OUT/'actual-partial-schema-observations.json').write_text(json.dumps(dict(
        exit_code=result.returncode, observations=rows,
        stderr=result.stderr.decode(errors='replace')[-3000:]), indent=2)+'\n')
    if result.returncode or not rows or rows[-1].get('status') != 'ACTUAL_TWO_BANK_SCHEMA_LINEAGE_AND_RECOVERED1216_INDEX_CHECK_PASSED':
        raise RuntimeError('Actual schema check failed; inspect preserved aggregate observation')
    (OUT/'actual-partial-schema-check.json').write_text(json.dumps(rows[-1], indent=2)+'\n')
    print(json.dumps(rows[-1], sort_keys=True))

if __name__ == '__main__':
    main()
