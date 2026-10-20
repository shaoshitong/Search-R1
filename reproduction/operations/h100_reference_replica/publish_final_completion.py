"""Publish small, actually accepted A/B results; no remote access or training."""
import datetime
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REPORT = REPO / 'reproduction/reports/h100-prefix5-ab200-20261018'

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def main():
    a_path = ROOT / 'work/prefix5-ab-final-qa-cpu-assigned-A-status-20261020T112221446831Z.json'
    b_path = ROOT / 'work/prefix5-ab-final-qa-cpu-assigned-B-status-20261020T111859051002Z.json'
    accepted = {'A': json.loads(read(a_path)['stdout'])['A'], 'B': json.loads(read(b_path)['stdout'])['B']}
    closures = read(ROOT / 'work/prefix5-ab-final-completion-small-proof.json')['arms']['B']
    b_closure = closures['B_closure']
    a_exit = read(ROOT / 'work/prefix5-ab-A-final-owned-existing-proof.json')['arms']['A']
    normal = read(ROOT / 'work/prefix5-ab-final-normal-proof-index.json')['arms']['B']
    qa512 = read(ROOT / 'work/prefix5-ab-final-QA512-history-index.json')['arms']['B']
    arms = read(ROOT / 'work/prefix5-ab-assigned-arms.json')
    history = read(REPORT / 'progressive-latest-actual/PER-ITERATION.json')
    sources = dict(accepted['A']['result']['actual_source_counts'])
    assert sum(sources.values()) == 51713 and len(sources) == 7
    for arm, point in accepted.items():
        v = point['result']
        assert v['status'] == 'FINAL51713_ORIGINAL_OUTPUT_SCORE_READBACK_PASS'
        assert v['actual_qa_count'] == v['actual_unique_qids'] == 51713
        assert v['actual_full_ordinal_coverage'] and v['actual_source_counts'] == sources
        assert v['actual_worker_exit_codes'] == {str(i): 0 for i in range(4)}
        assert v['run_id'] == arms[arm]['run_id']
        assert [r['outer'] for r in normal[arm]] == [5, 50, 100, 150, 200]
        assert [r['outer'] for r in qa512[arm]] == [5, 50, 100, 150, 200]
        rows = history['current'][arm]['rows']
        assert [r['outer'] for r in rows] == list(range(1, 201))
        assert all(r['selected']['questions'] == 512 for r in rows)
    assert not history['unobserved_values_fitted']
    assert a_exit['actual_native_returncode'] == 0 and a_exit['complete_outer'] == 200
    assert len(a_exit['checks']) == 2 and all(not x['native_same_birth'] and not x['remaining_owned_births'] and not x['owned_gpu_rows'] for x in a_exit['checks'])
    assert b_closure['actual_native_returncode'] == 0 and b_closure['no_outer201']
    assert len(b_closure['owned_release_checks']) == 2 and all(not x['native_same_birth'] and not x['supervisor_same_birth'] and not x['remaining_owned_births'] and not x['owned_gpu_pid_rows'] for x in b_closure['owned_release_checks'])
    results = {}
    for arm, proof in accepted.items():
        results[arm] = dict(run_id=arms[arm]['run_id'], base=arms[arm]['base'], frozen_runtime_commit=arms[arm]['commit'],
            complete_outers=200, actor_actual_adam_per_rank=normal[arm][-1]['actor_adam'],
            probe_actual_adam_per_rank=normal[arm][-1]['probe_adam'], qa512_200=qa512[arm][-1],
            final51713=proof['result']['aggregate'], final_qa_proof=dict(path=proof['proof'], sha256=proof['proof_sha256']),
            original_output_full_sha_cpu_rescore_pass=True, exact_qids_and_ordinals=True,
            exact_seven_source_counts=sources, four_final_workers_exit0=True, actual_native_exit0=True)
    differences = {k: (results['B']['final51713'][k] - results['A']['final51713'][k]) * 100 for k in ('em', 'f1')}
    result = dict(status='BOTH200_AND_FINAL51713_ACCEPTED', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        arms=results, B_minus_A_percentage_points=differences, normal_checkpoints=normal,
        qa512_history=qa512, B_current_execution_retention_and_release=b_closure,
        A_current_execution_release=a_exit, A_milestone_retention=read(REPORT / 'A-normal200-final51713/RETENTION.json'),
        per_iteration_data='progressive-latest-actual/PER-ITERATION.csv', unobserved_values_fitted=False,
        interpretation='B also changes prefix length and outcome-assisted admission/reranking after50; this is not an isolated EMA ablation.',
        A_historical_actor_skips=16, independent_physical_backup_claimed=False,
        historical_source11_kernel_residues='Separately documented; no claim that other/historical executions or the whole shared root cgroup were cleared.',
        models_optimizer_trajectories_full_predictions_stay_remote=True,
        peer_resource_release_notice_delivered=True, no_201=True)
    write(REPORT / 'FINAL-RESULTS.json', result)
    write(REPORT / 'NORMAL-CHECKPOINTS.json', normal)
    write(REPORT / 'QA512-HISTORY.json', qa512)
    b_report = REPORT / 'B-normal200-final51713'
    write(b_report / 'RESULT.json', results['B'])
    write(b_report / 'RETENTION-AND-RELEASE.json', b_closure)
    write(b_report / 'PROGRESS.json', dict(status='COMPLETED_AND_ACCEPTED', complete_outer=200, final_questions=51713,
        original_output_full_sha_cpu_rescore_pass=True, actual_native_exit0=True, current_owned_release_pass=True))
    (b_report / 'README.md').write_text('# B：200轮及最终51713题已完成\n\n固定512题：EM49.0234375%，F157.3902529762%。最终51713题：EM48.1484346296%，F155.1775522142%。四个原始评测进程均exit0，全部七集、题目ID及顺序覆盖通过，原始gzip完整SHA及逐题CPU重算分数通过。\n\nActor4000、raw/EMA probe2760实际Adam/rank。完整第200轮状态及5/50/100/150保存点已做独立全SHA和CPU状态读回；既有通过点没有重复大文件检查。当前native实际exit0，当前自有birth/GPU两次释放核对通过。全部200轮所选题目/完整轨迹和保存点原件留远端，Mac仅小摘要。见RESULT.json和RETENTION-AND-RELEASE.json。\n\n历史source11 NFS退出残留另记；不声称清理所有历史/外来GPU进程，也不声称完成独立物理备份。未执行201轮。\n')
    table = ['| 数据集 | 题数 | A EM | A F1 | B EM | B F1 |', '|---|---:|---:|---:|---:|---:|']
    for source, count in sources.items():
        av, bv = results['A']['final51713']['per_source'][source], results['B']['final51713']['per_source'][source]
        table.append(f"| {source} | {count} | {av['em']*100:.2f}% | {av['f1']*100:.2f}% | {bv['em']*100:.2f}% | {bv['f1']*100:.2f}% |")
    text = '# A/B正式200轮与51713题最终结果\n\n两组均完成200轮，原生作业实际exit0；全部最终题目、原始评分及当前作业自有资源释放已验收。未执行201轮。\n\n| 组别 | 200轮固定512题EM | 固定512题F1 | 最终51713题EM | 最终F1 |\n|---|---:|---:|---:|---:|\n'
    for arm in ('A', 'B'):
        v = results[arm]
        text += f"| {arm} | {v['qa512_200']['em']*100:.2f}% | {v['qa512_200']['f1']*100:.2f}% | {v['final51713']['em']*100:.2f}% | {v['final51713']['f1']*100:.2f}% |\n"
    text += f"\nB高出{differences['em']:.2f}个EM百分点、{differences['f1']:.2f}个F1百分点。A冻结warmup探针；B在线均衡拟合raw探针、EMA筛选，并从50轮后采用20/40/80-token渐进前缀及768/1024/1536组结果辅助筛选。因此收益不能单独归因于EMA，额外采样预算也不同。A有16次已记录actor skip（最终3984/rank），B为4000/rank；B probe/EMA2760/rank。\n\n"
    text += '\n'.join(table) + '\n\n'
    text += '全部5/50/100/150/200完整保存点已通过独立全文件SHA与实际CPU状态读回。NORMAL-CHECKPOINTS.json保留原证明路径/SHA，未重复已完成的大文件读取。全部计划512题验证已从原四rank收据重新核对固定题目ID/顺序/汇总；QA512-HISTORY.json记录原始结果。B在50恢复时额外验证的结果不替换源50主结果，两者差异记录在历史报告。\n\n'
    text += '最终51713题两组均核对七个来源的精确数量、51713唯一qid及全部ordinal；四个原始gzip逐文件完整SHA并基于真实token、原gold及冻结评分代码CPU重算EM/F1，与原汇总完全一致。未重新生成评测。原始结果、权重、优化器、轨迹均保留远端。\n\n'
    text += '当前A/B实例的实际退出及自有资源释放均已证明；历史source11 NFS退出残留仍是单独维护记录，不属于当前实例，未冒称整个节点或共享根cgroup已清空。没有操作其他实验或foreign任务。本次只声称已核验的远端留存，不声称额外独立物理备份已完成。\n\n'
    text += '逐轮真实数据：[PER-ITERATION.csv](progressive-latest-actual/PER-ITERATION.csv)。两组各1–200轮，无拟合缺失值；周期轮使用结果辅助选择，与普通轮分开标记。完整验收证据见FINAL-RESULTS.json与COMPLETION-AUDIT.json。\n'
    (REPORT / 'FINAL-RESULTS.md').write_text(text)
    evidence = dict(normal_checkpoint_index=str(ROOT / 'work/prefix5-ab-final-normal-proof-index.json'),
        qa512_index=str(ROOT / 'work/prefix5-ab-final-QA512-history-index.json'),
        final_A_cpu_status=str(a_path), final_B_cpu_status=str(b_path),
        current_B_closure=str(ROOT / 'work/prefix5-ab-final-completion-small-proof.json'),
        current_A_exit=str(ROOT / 'work/prefix5-ab-A-final-owned-existing-proof.json'))
    checks = [
        ('Two genuinely complete200 trajectories', 'PASS', 'PER-ITERATION has contiguous unique1..200 per arm; original journals/committed joints and final CPU clocks'),
        ('Actual final checkpoint model/masterAdam/probe/EMA/scheduler/RNG/pool', 'PASS', normal),
        ('Scheduled QA512 at5/50/100/150/200 with original canonical IDs and worker aggregates', 'PASS', qa512),
        ('Final51713 exact seven sources, qids, ordinals and all original worker exits', 'PASS', {a:v['result']['actual_source_counts']for a,v in accepted.items()}),
        ('Original final outputs complete SHA and true token/gold scorer CPU rescore', 'PASS', {a:dict(path=v['proof'],sha256=v['proof_sha256'])for a,v in accepted.items()}),
        ('Actual native exit0/current-owned release, not local observer exit', 'PASS', dict(A=a_exit, B=b_closure['owned_release_checks'])),
        ('Normal milestone remote retention; original outputs stay remote; no201', 'PASS', dict(A=read(REPORT/'A-normal200-final51713/RETENTION.json'),B=b_closure)),
        ('Result publication and full Git object readback', 'POST_COMMIT_RECEIPT_REQUIRED', 'The authoritative post-push accepted audit is /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/prefix5-ab-final-publication-accepted.json; it binds this report commit and independent full Git SHA. This static report necessarily precedes its own commit.')]
    audit = dict(objective='持续监督，别停，直到完成200iter训练和51713题目评测，如果退出了那么及时resume',
        scope='Current approved A/B formal200 including its retained protocol, final QA and current-owned cleanup. Superseded experiments and peer diverse10to5 are not this goal.',
        requirements=[dict(requirement=r,status=s,evidence=e)for r,s,e in checks],
        evidence_files={k:dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for k,p in evidence.items()},
        independent_physical_backup_asserted=False, historical_kernel_residue_not_claimed_released=True,
        no_new_forward_or_training_for_audit=True, no_model_or_full_output_downloaded=True)
    write(REPORT / 'COMPLETION-AUDIT.json', audit)
    p = REPORT / 'CURRENT-RUN-HANDOFF.md'
    p.write_text(p.read_text() + '\n## Final completion evidence (2026-10-20)\n\nBoth formal arms completed200 and exact51713, current nativeexit0/current-owned release accepted. B final EM48.1484346296%/F155.1775522142%; source CPU proof e2df517ead79161408b1a1d738abd2cc38616e805de53a58118d52936100bf64. B final retention/current-release proof40c6fa872319779422dba7c762b2ac41e58dd20292a5f0f2466f6d811fde43da; two current-owned checks pass, all200 selected original payloads retained, no201. All ten normal checkpoint proofs and ten QA512 records were read as small metadata and identities/aggregates checked, without repeating any large tensor hash/load. No active local training observer. The separately owned peer diverse10to5 still runs and received the release notice via actual local host; its automation is not ours and must not be paused/deleted. Historical NFS kernel residues remain separate; do not invent a current-goal failure or claim all-node release. See FINAL-RESULTS.md/json and COMPLETION-AUDIT.json. Only pending task is final publication/readback then actual API goal completion; never resume/retrain a completed200 actor.\n')
    p = REPORT / 'README.md'
    p.write_text('> Final accepted result: [FINAL-RESULTS.md](FINAL-RESULTS.md). Both arms200 and final51713 are complete; historical progress below is time-stamped history.\n\n' + p.read_text())
    print(json.dumps(dict(status=result['status'], report=str(REPORT/'FINAL-RESULTS.md'), differences=differences)))

if __name__ == '__main__':
    main()
