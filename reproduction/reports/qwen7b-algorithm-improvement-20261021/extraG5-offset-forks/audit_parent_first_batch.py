"""Single *sealed* matched6528 offset5 batch audit; all large bytes stay remote."""
import hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
P='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
COLLECTOR=P+'/matched-g10-collector-code-20261021/code-2b293d6e6a0b'
CODE=r'''
import pathlib,json,gzip,hashlib,time,inspect
from reproduction.operations.qwen7b_warmup.extra_g5_prepare import validate_config,ROLE,REVISION
from reproduction.prefix_selection200.rollout import seed_for,strict_label
from reproduction.prefix20_online.pool import canonical
from transformers import AutoTokenizer
P=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
b=P/'algorithm-improvement-20261021/training-extraG5-matched6528-20261021/materialized'
started=time.monotonic()
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(ref):
 p=pathlib.Path(ref['path']);assert p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],str(p)
 return p
def rows(p,compressed=False):
 opener=gzip.open if compressed else open
 with opener(str(p),'rt') as f:return [json.loads(line) for line in f]
assert sha(b/'config.json')=='3957642964d692329b98b2de216c9b7cb2ebe9866a667283e0055bb36223aa67'
c=validate_config(read(b/'config.json'));assert c['source_commit']=='2b293d6e6a0be5a8b99f976b3ac0c2b15e19ee40'
for name,digest in c['code_files_sha256'].items():assert sha(pathlib.Path(c['code_root'])/name)==digest,name
assert sha(b/'inputs/questions.json')=='265ce78e1e19e87263256348ef7fa54bf09987f7be65e331b3541e427a11f703'
qplan=read(b/'inputs/questions.json');old={g['qid']:g for g in qplan['groups']}
batchdir=b/'sampling/shard-00/batches/batch-000000';receipt=read(batchdir/'complete.json')
assert receipt['run_id']==c['run_id'] and receipt['actor_optimizer_updates']==0
sourcepath=b/'inputs/shard-00/batch-000000.json';assert receipt['input_sha256']==sha(sourcepath)
source=read(sourcepath);inputs={r['qid']:r for r in source['rows']};assert len(inputs)==128
assert list(inputs)==[g['qid'] for g in qplan['groups'][::4]][:128]
prefix,full=receipt['prefix'],receipt['full'];fresh=[]
for phase,keys in ((prefix,('states','prefix','actor_hidden')),(full,('full','own_em'))):
 assert phase['run_id']==c['run_id'] and phase['actor_version']=='official7b-'+REVISION
 assert phase['sampling_attempt_id']==c['sampling_attempt_id']
 assert {k:phase[k] for k in c['original_protocol_contract']}==c['original_protocol_contract']
 for key in keys:
  ref=phase[key];p=pathlib.Path(ref['path']);assert str(p.resolve()).startswith(str(batchdir.resolve())+'/')
  checked(ref);fresh.append(dict(key=key,**ref))
assert prefix['sampling_contract']==full['sampling_contract'] and prefix['interaction_protocol']==full['interaction_protocol']
feature=read(prefix['actor_hidden']['path']);assert feature['status']=='LOSSLESS_NATIVE_FEATURES_COMPLETE'
assert feature['feature_layer']==26 and feature['prefix_tokens']==20
for name,item in feature['files'].items():checked(item);fresh.append(dict(key='native-feature:'+name,**item))
prefixrows=rows(prefix['prefix']['path'],True);fullrows=rows(full['full']['path'],True);emrows=rows(full['own_em']['path'])
assert len(prefixrows)==len(fullrows)==len(emrows)==640
pre={r['trajectory_id']:r for r in prefixrows};actual={r['trajectory_id']:r for r in fullrows};ems={r['trajectory_id']:r for r in emrows}
assert len(pre)==len(actual)==len(ems)==640 and set(pre)==set(actual)==set(ems)
tokenizer=AutoTokenizer.from_pretrained(c['model_dir'],local_files_only=True,trust_remote_code=False)
byqid={};seedsets={};seed_count=0;em_mismatches=0;prefix_lengths={};retrievals=0
for tid,r in pre.items():
 f=actual[tid];e=ems[tid];qid=r['qid'];assert qid in inputs
 assert r['run_id']==f['run_id']==c['run_id'] and r['actor_version']==f['policy_snapshot_id']=='official7b-'+REVISION
 assert r['response_index'] in list(range(5,10)) and f['response_index']==r['response_index']
 assert r['split']==f['split']=='train' and f['qid']==e['qid']==qid
 assert r['question_hash']==f['question_hash']==inputs[qid]['question_hash']==old[qid]['question_hash']
 assert tid not in old[qid]['trajectory_ids'] and inputs[qid]['original_trajectory_ids']==old[qid]['trajectory_ids']
 assert r['initial_prompt_token_ids']==f['initial_prompt_token_ids']==inputs[qid]['initial_prompt_token_ids']
 assert f['prefix_sha256']==e['prefix_sha256']==r['prefix_sha256']
 assert r['future_information_included'] is False and 0<len(r['policy_token_ids'])<=20
 assert len(f['response_token_ids'])==len(f['response_info_mask'])
 assert r['policy_token_ids']==[x for x,m in zip(f['response_token_ids'],f['response_info_mask']) if m][:len(r['policy_token_ids'])]
 assert f['behavior_turns'] and f['sampling_journal']
 label=e['strict_em'];assert type(label) is int and label in (0,1) and f['strict_em']==label
 recomputed=strict_label(tokenizer,f['response_token_ids'],f['response_info_mask'],f['reward_model']['ground_truth'])
 em_mismatches+=int(label!=recomputed)
 original_seeds=set(qplan['original_native_seeds_by_qid'][qid]);own=seedsets.setdefault(qid,set())
 for chunk in f['sampling_journal']:
  expected=seed_for(c['run_id'],1,tid,chunk['logical_turn_index'],chunk['native_chunk_index'],c['seed'],sampling_attempt_id=c['sampling_attempt_id'])
  assert chunk['seed']==chunk['sampling']['seed']==expected
  assert expected not in original_seeds and expected not in own,'Actual seed collision within same question/original G5'
  own.add(expected);seed_count+=1
 byqid.setdefault(qid,[]).append((r['response_index'],tid,label));prefix_lengths[str(len(r['policy_token_ids']))]=prefix_lengths.get(str(len(r['policy_token_ids'])),0)+1
 retrievals+=f.get('retrieval_calls',0)
assert em_mismatches==0 and set(byqid)==set(inputs)
hist=[0]*6
for qid,group in byqid.items():
 assert sorted(x[0] for x in group)==list(range(5,10)) and len({x[1] for x in group})==5
 hist[sum(x[2] for x in group)]+=1
import search_g1.signals,verl.utils.reward_score.qa_em
scoring_sources=[fp(inspect.getfile(v)) for v in (strict_label,search_g1.signals,verl.utils.reward_score.qa_em)]
result=dict(status='FIRST_SEALED_EXTRA_G5_BATCH_SHA_OWN_EM_NATIVE_SEED_CONTRACT_VERIFIED',job_id='743115_0',batch='batch-000000',shard=0,questions=128,trajectories=640,
 run_id=c['run_id'],sampling_attempt_id=c['sampling_attempt_id'],response_indices=list(range(5,10)),collector_commit=c['source_commit'],receipt=fp(batchdir/'complete.json'),input=fp(sourcepath),
 fresh_SHA_files=len(fresh),fresh_SHA_bytes=sum(r['bytes'] for r in fresh),fresh_file_proofs=fresh,
 model_tensor_loaded=False,tokenizer_loaded_on_remote_CPU_only=True,tokenizer_dir=c['model_dir'],strict_EM_recomputed=640,strict_EM_mismatches=em_mismatches,own_label_histogram_K0_to5=hist,
 mixed_questions=sum(hist[1:5]),mixed_fraction=sum(hist[1:5])/128,correct_trajectories=sum(k*n for k,n in enumerate(hist)),
 first20_policy_tokens_equal_original_full_continuation=True,future_information_in_prefix=False,prefix_length_histogram=prefix_lengths,
 actual_sampling_journal_seed_chunks=seed_count,actual_seed_collisions_same_question_or_original_G5=0,seed_derivation_recomputed=True,
 same_prompt_original5_own_label_and_source_protocol_preserved=True,retrieval_calls=retrievals,actor_updates=0,
 full_shard_complete_claim=False,future_G20_cross_phase_seed_check_not_yet_possible=True,strict_scoring_source_files=scoring_sources,
 elapsed_seconds=time.monotonic()-started,observed_epoch=time.time(),large_payloads_kept_remote=True)
print(json.dumps(result))
'''

def main():
 spec=importlib.util.spec_from_file_location('first_batch_audit_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 command='QWEN7B_CODE_ROOT='+shlex.quote(COLLECTOR)+' CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash '+shlex.quote(COLLECTOR+'/reproduction/operations/qwen7b_warmup/runtime.sh')+' -c '+shlex.quote(CODE)
 assert len(command.encode())<16000
 r=c.run(command,account='mou176',source_address=address,timeout=160)
 values=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
 if r.returncode or not values:raise RuntimeError(r.stderr.decode(errors='replace')[-2500:]+' '+r.stdout.decode(errors='replace')[-3000:])
 result=values[-1];result['operator_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();result['operator_command_bytes']=len(command.encode())
 (Path(__file__).parent/'first-parent-batch-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 (CONTROL/'outputs/qwen7b-warmup-20261019/extraG5-first-matched6528-batch-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
