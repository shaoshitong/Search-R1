import argparse,hashlib,json,os,pathlib,subprocess,tempfile,time
parser=argparse.ArgumentParser();parser.add_argument('--local-only',action='store_true');options=parser.parse_args()
repo=pathlib.Path(__file__).resolve().parents[3]
branch='codex/q7b-native-joint-root-20261021'
ref='refs/heads/'+branch
def git(*args,env=None):
 return subprocess.check_output(['git',*args],cwd=repo,env=env,text=True).strip()
parent=git('rev-parse',ref)
files=[pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('joint_mixed_direct_labelblind.py','direct_update_labelblind.py','labelblind_question_g5_sampler.py','deploy_mixed_direct_e1_uniform_g5_weighted.py','prepare_frozen_final4096_generation.py','deploy_frozen_final4096_scoring.py','score_frozen_final4096.py','test_score_frozen_final4096.py','accept_online_mixed_pilot_checkpoints.py','deploy_online_pilot_CPU_acceptance.py')]
report=pathlib.Path(__file__).resolve().parent
for folder in ('mixed-direct-e1-unique-g5','online-layer-prefix','online-pilot-CPU-acceptance','uniform-g5-control','unique-vs-labelblind-g5','frozen-final4096-preparation','frozen-final4096-scoring','live-observations'):
 for p in (report/folder).rglob('*'):
  if p.is_file() and p.suffix in ('.py','.md','.json','.jsonl','.txt') and p.stat().st_size<600000:files.append(p.relative_to(repo))
files.extend(pathlib.Path(__file__).resolve().parent.relative_to(repo)/n for n in ('observe-live.py','publish-goal-progress.py','read-current-queue-forecast.py','raise-waiting-current-jobs-medium.py','raise-waiting-current-jobs-third-tier.py','LATEST-ACTIVE-EXPERIMENTS.json'))
files=sorted(set(p for p in files if (repo/p).is_file()))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('compare_viewed2048_pilot_depth.py','deploy_compare_viewed2048_pilot_depth.py','deploy_online_l21p80_factorial.py','online_mixed_depth_expansion.py','deploy_online_matched_depth_expansion.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('compare_viewed2048_expanded_depth.py','deploy_compare_viewed2048_expanded_depth.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('natural_g5_online_control.py','deploy_natural_g5_online_control.py','packed_variance_probe.py','online_packed_variance.py','deploy_online_packed_variance.py','test_packed_variance_probe.py'))
files.append(pathlib.Path('reproduction/operations/qwen7b_warmup/test_packed_variance_full_schema.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('audit_natural_g5_online_checkpoint.py','compare_natural_g5_online_matched.py','stage_natural_g5_postrun_audits.py','accept_packed_variance_checkpoint.py','deploy_packed_variance_CPU_acceptance.py','online_packed_variance_scaling.py','deploy_full169028_packed_variance.py','accept_full169028_packed_variance.py','deploy_full169028_packed_CPU_acceptance.py'))
files.extend(p.relative_to(repo) for p in (report/'viewed2048-pilot-depth-comparison').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<600000)
files.extend(p.relative_to(repo) for p in (report/'online-matched-depth-expansion').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<600000)
files.extend(p.relative_to(repo) for p in (report/'viewed2048-expanded-depth-comparison').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<600000)
for folder in ('natural-g5-online-8192','packed-variance','packed-variance-M3to10-8192','packed-variance-CPU-acceptance','login-incident','packed-variance-full169028','packed-variance-full169028-CPU-acceptance','historical-all-experiments','online-matched-prefix80-expansion'):
 files.extend(p.relative_to(repo) for p in (report/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<600000)
files.extend(report.relative_to(repo)/n for n in ('read-current-metrics-table.py','read-all-pilot-epochs-metrics.py','CURRENT-METRICS-TABLE.md','CURRENT-METRICS-TABLE.json'))
files.append(report.relative_to(repo)/'historical-all-experiments/ALL-HISTORICAL-RESULTS.json')
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('compare_full169028_packed_variance.py','deploy_compare_full169028_packed_variance.py'))
files.extend(p.relative_to(repo) for p in (report/'viewed2048-full169028-packed-comparison').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<1000000)
files.append(pathlib.Path('DERIVATION_PACKAGE.md'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('group_topk_losses.py','test_group_topk_losses.py','online_packed_matched_objective.py','deploy_matched_packed_objective.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('detached_score_window.py','test_detached_score_window.py','score_window_audit.py','test_score_window_audit.py','online_packed_window_topk.py','deploy_packed_window_topk.py','test_packed_window_runtime_cpu.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('online_packed_variance_seeded.py','deploy_seeded_full169028_Z.py','test_online_packed_variance_seeded.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('accept_matched_window_checkpoint.py','deploy_matched_window_CPU_acceptance.py','test_matched_window_acceptance_metadata.py'))
for folder in ('group-topk-alignment','matched-packed-objective','P80-terminal-CPU-acceptance','P80-boundary-handoff','detached-score-window','score-window-audit','global64-window-topk','leading-Z-replication','packed-Z-E8-trajectory-permutation-audit','seeded-full169028-Z','matched-window-CPU-acceptance','seeded-current64-topk','seeded-matched-MSE-M5','seeded-window2018-topk','seeded-window2018-topk-v2','leading-P80-replications'):
 files.extend(p.relative_to(repo) for p in (report/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<1000000)
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('online_packed_window_topk_seeded.py','online_packed_window_topk_seeded_v2.py','deploy_seeded_window2018_topk_v2.py','accept_seeded_window2018_v2_checkpoint.py','accept_seeded_matched_window_checkpoint.py','online_packed_matched_objective_seeded.py','online_packed_full169028_MSE_P80.py','deploy_full169028_MSE_P80.py','test_full169028_MSE_P80_cpu.py','prepare_full169028_policy80_physical_v2.py','deploy_full169028_policy80_physical_v2.py','crosscheck_full169028_policy80_P20.py','deploy_full169028_policy80_physical_P20_crosscheck.py','score_frozen_architectures_final4096.py','deploy_frozen_architectures_final4096.py','test_frozen_architectures_final4096.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('accept_full169028_L15P80_MSE_checkpoint.py','deploy_full169028_L15P80_MSE_CPU_acceptance.py','test_full169028_L15P80_MSE_acceptance_metadata.py','score_frozen_full169028_MSE_final4096.py','test_frozen_full169028_MSE_scorer_admission.py'))
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('accept_full169028_Z_seed2_checkpoint.py','deploy_full169028_Z_seed2_CPU_acceptance.py','fullZ_seed2_terminal_CPU_coordinator.py','deploy_fullZ_seed2_terminal_CPU_coordinator.py','full169028_MSE_metadata_observer.py','deploy_full169028_MSE_metadata_observer.py'))
files.extend(report.relative_to(repo)/n for n in ('CURRENT-DEVELOPMENT-DECISION.md','CURRENT-DEVELOPMENT-DECISION.json'))
for folder in ('full169028-P80-preparation','full169028-P80-source-inventory','full169028-P80-physical-v2','full169028-P80-physical-P20-audit','viewed-label-ownEM-audit','viewed2048-two-seed-P80-paired-comparison','viewed2048-current64-window2018-sameE10-comparison','fullZ-seed2-live-20261023','full169028-Z-seed2-CPU-acceptance','full169028-L15P80-MSE-ownG5','full169028-L15P80-MSE-CPU-acceptance','full169028-L15P80-MSE-metadata-observer','full169028-MSE-metadata-observer','fullMSE-750190-actual-E1-sampler','full169028-L15P80-permutation-audit','current-development-decision-sources','seed2-window2018-v2-CPU-acceptance','seed2-matched-window-CPU-acceptance'):
 files.extend(p.relative_to(repo) for p in (report/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.jsonl','.txt') and p.stat().st_size<1000000)
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('score_selected_fixed_trainB.py','deploy_selected_fixed_trainB.py','test_selected_fixed_trainB.py','deploy_full169028_frozen_final4096_scoring.py','test_frozen_full169028_MSE_final4096.py','packed_full169028_trajectory_permutation_audit.py','deploy_packed_full169028_permutation_audit.py','test_packed_full169028_trajectory_permutation_audit.py'))
for folder in ('selected-state-fixed-trainB','frozen-full169028-MSE-final4096-scoring'):
 files.extend(p.relative_to(repo) for p in (report/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<1000000)
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('select_compare_full169028_P80_MSE.py','test_full169028_MSE_family_selection.py','deploy_full169028_P80_MSE_family_analysis.py'))
files.extend(p.relative_to(repo) for p in (report/'full169028-P80-MSE-family-selection').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<1000000)
files.extend(pathlib.Path('reproduction/operations/qwen7b_warmup')/n for n in ('fullP80_MSE_terminal_CPU_coordinator.py','deploy_fullP80_MSE_terminal_CPU_coordinators.py','test_fullP80_MSE_terminal_coordinator.py','bind_submit_selected_trainB_once.py','deploy_selected_trainB_root_control.py','test_bind_submit_selected_trainB_once.py'))
files.extend(p.relative_to(repo) for p in (report/'fullP80-MSE-terminal-CPU-coordinators').rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and p.stat().st_size<1000000)
files=sorted(set(p for p in files if (repo/p).is_file()))
fd,index=tempfile.mkstemp(prefix='q7b-root-publish-');os.close(fd);os.unlink(index)
env=dict(os.environ,GIT_INDEX_FILE=index)
try:
 git('read-tree',parent,env=env);git('add','--',*[str(p)for p in files],env=env)
 tree=git('write-tree',env=env)
 if tree==git('rev-parse',parent+'^{tree}'):commit=parent
 else:
  commit=subprocess.check_output(['git','commit-tree',tree,'-p',parent],cwd=repo,env=env,text=True,input='Preserve two-GPU pilot progress and isolated G5 control and sealed production\n').strip()
  git('update-ref',ref,commit,parent)
 remote=None
 if not options.local_only:
  git('push','origin',ref+':refs/heads/'+branch)
  remote=git('ls-remote','origin',ref).split()[0];assert remote==commit
 result=dict(time=time.time(),branch=branch,commit=commit,remote_sha=remote,push_verified=remote==commit,local_only=options.local_only,files=[dict(path=str(p),bytes=(repo/p).stat().st_size,sha256=hashlib.sha256((repo/p).read_bytes()).hexdigest())for p in files])
 report_name='root-pending-publication.json'if options.local_only else'root-current-publication.json'
 (report/report_name).write_text(json.dumps(result,sort_keys=True)+'\n')
 print(json.dumps(dict(commit=commit,remote_sha=remote,push_verified=remote==commit,local_only=options.local_only,files=len(files))))
finally:
 if os.path.exists(index):os.unlink(index)
