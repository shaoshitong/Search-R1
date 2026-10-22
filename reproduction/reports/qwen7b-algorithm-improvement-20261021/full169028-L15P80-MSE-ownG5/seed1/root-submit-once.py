"""Root-only single submission after actual physical-source and P20 admission."""
EXPECTED={'status': 'FULL169028_L15P80_MSE_OWNG5_SOURCE_STAGED_NOT_TRAINING_READY', 'publication': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3/source-publication.json', 'bytes': 2020, 'sha256': 'e6832ea245d9467bdf8abef0585c7511372a057b7fc058c2298f816021e61490'}, 'entry': {'path': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3/training.sbatch', 'bytes': 1829, 'sha256': 'd065167ea206cc2b894ab1a9becc4d2c7e283b5a81aa47c7092f42184d5e91c8'}, 'fixture_expected': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-physical-ownG10-20261023-v2/fixture', 'fixture_expected_status': 'FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_READY', 'training': {'questions': 169028, 'epochs': 10, 'updates_per_epoch': 1321, 'total_updates': 13210, 'global_batch': 128, 'local_batch': 64, 'micro': 8, 'accumulation': 8, 'tail_global_batch': 68, 'tail_local_batch': 34, 'tail_micro_counts': [8, 8, 8, 8, 2], 'fixed_m': 5, 'layer': 15, 'prefix': 80, 'head_max_positions': 2048, 'LR': 5e-06, 'EMA_decay': 0.9995, 'init_seed': 9137, 'train_plan_seed': 2026102227, 'diagnostic_seed': 2026102228, 'fresh_official_block': True, 'pilot_resume': False, 'raw_to_EMA_continuous': True}, 'resources': {'partition': 'acd_u', 'qos': 'normal', 'GPUs': 2, 'CPUs': 12, 'memory_GiB': 128, 'time': '12:00:00'}, 'replica': 'seed1', 'evaluation_protocol': {'primary': 'viewed_independent2048_ownG5_Top512_true_Mixed', 'epoch_and_raw_or_EMA_selected_using_seed1_dev_only': True, 'same_selected_epoch_state_applied_to_seed2': True, 'seed2_not_used_for_selection': True, 'final4096_read': False}, 'actual_SBATCH_full_CLI_parser_passed': True, 'CPU_import_and_entry_syntax_passed': True, 'fixture_CPU_acceptance_required_before_root_submission': True, 'P20_alignment_required_before_root_submission': True, 'actual_physical_execution_and_seal_source_required_before_root_submission': True, 'Slurm_submitted': False, 'final4096_read': False}
import fcntl,hashlib,json,pathlib,re,subprocess,time

def read(path):return json.loads(path.read_bytes())
def proof(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def run(args):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
    if p.returncode:raise RuntimeError(p.stdout[-800:]+p.stderr[-1200:])
    return p.stdout.strip()
base=pathlib.Path(EXPECTED['entry']['path']).parent
lock=(base/'root-submit.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
entry=base/'training.sbatch';publication=base/'source-publication.json'
assert proof(entry)==EXPECTED['entry'] and proof(publication)==EXPECTED['publication']
pub=read(publication);ready_path=base/'ready.json';receipt=base/'root-submission.json';intent=base/'root-submit-intent.json'
if receipt.exists():
    saved=read(receipt)
    assert saved['entry']==proof(entry) and saved['source_commit']==pub['commit']
    print(json.dumps(saved));raise SystemExit(0)
if not ready_path.exists():
    print(json.dumps(dict(status='WAITING_REAL_ACCEPTED_INPUT_NO_SUBMISSION',base=str(base),time=time.time(),Slurm_submitted=False)));raise SystemExit(0)
ready=read(ready_path)
assert ready['status']=='FULL169028_L15P80_MSE_OWNG5_ACCEPTED_FIXTURE_READY_NOT_SUBMITTED'
assert ready['entry']==proof(entry) and ready['publication']==proof(publication)
fixture=pathlib.Path(EXPECTED['fixture_expected'])
assert ready['fixture']==proof(fixture/'fixture-complete.json') and ready['independent_acceptance']==proof(fixture/'accepted.json')
accepted=read(fixture/'accepted.json')
assert ready['actual_physical_execution_and_seal_source_exact'] is True and accepted['actual_physical_execution_and_seal_source_exact'] is True
assert ready['fresh_full_P20_prefix_prompt_and_own_EM_exact'] is True and accepted['fresh_full_P20_prefix_prompt_and_own_EM_exact'] is True
assert ready['physical_source_contract']==proof(pathlib.Path(accepted['physical_source_contract']['path']))
assert ready['P20_alignment']==proof(pathlib.Path(accepted['P20_alignment']['path']))
assert ready['training_questions']==169028 and ready['own_G10_trajectories']==1690280
name=re.search(r'^#SBATCH --job-name=(.+)$',entry.read_text(),re.M).group(1)
rows=run(['squeue','-h','-u','mou176','-n',name,'-o','%i']).splitlines()
if rows:
    assert len(rows)==1
    job=rows[0]
    details=run(['scontrol','show','job',job,'-o'])
    assert 'UserId=mou176(' in details and 'Command='+str(entry) in details
else:
    if intent.exists():raise RuntimeError('Prior submission outcome unknown: inspect this exact name/accounting before retrying.')
    assert not(base/'controller.json').exists() and not(base/'latest.json').exists()
    with intent.open('x')as f:
        json.dump(dict(time=time.time(),entry=proof(entry),ready=proof(ready_path),name=name),f,sort_keys=True);f.flush()
    job=run(['sbatch','--parsable','--exclude=ACD1-13',str(entry)]).split(';')[0]
    assert job.isdigit()
result=dict(time=time.time(),job_id=job,name=name,entry=proof(entry),ready=proof(ready_path),source_commit=pub['commit'],init_seed=EXPECTED['training']['init_seed'],train_plan_seed=EXPECTED['training']['train_plan_seed'],GPUs=2,partition='acd_u',QoS='normal',epochs=10,questions=169028,global_batch=128,updates_per_epoch=1321,optimizer_update_budget=13210,fresh=True,pilot_resume=False,primary_metric='development2048_ownG5_Top512_real_Mixed',selection='seed1_epoch_state_then_same_on_seed2',no_dependencies=True,final4096_read=False)
with receipt.open('x')as f:json.dump(result,f,sort_keys=True)
print(json.dumps(result))
