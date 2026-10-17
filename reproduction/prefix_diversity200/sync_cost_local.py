"""Observe the existing run, synchronize scalar evidence, and optionally publish.

No training submission, stop, model loading, or runtime edit is performed.
Published cost readers only write their own small remote audit-cache proofs.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from reproduction.h1008_qprobe.management_transport import remote
from .cost_reservation import collect as reservation_cost
from .refresh_local import refresh


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()


def sync(root, publish=False):
    root = Path(root).resolve()
    repo = Path(__file__).resolve().parents[2]
    work = root/'work/search-g1-prefix1024-diversity512-20261016'
    output = root/'outputs/search-g1-prefix1024-diversity512'
    evidence = repo/'reproduction/prefix_diversity200/control_evidence'
    config = read(work/'CURRENT_CONFIG.json')
    commit, branch = git(repo, 'rev-parse', 'HEAD'), git(repo, 'branch', '--show-current')
    if git(repo, 'ls-remote', 'origin', 'refs/heads/'+branch).split()[0] != commit:
        raise ValueError('Publish the current reader source before executing it')
    entry = 'reproduction/prefix_diversity200/sync_cost_local.py'
    if subprocess.check_output(['git', 'show', commit+':'+entry], cwd=repo) != Path(__file__).read_bytes():
        raise ValueError('This entry must match its published source')
    observation = refresh(root)
    previous = read(work/'COST_SNAPSHOT.json')
    earliest = {}
    for row in previous['complete_outer_rows'].values():
        earliest[row['base']] = min(earliest.get(row['base'], row['outer']), row['outer'])
    bases = sorted(earliest, key=earliest.get)
    if config['base'] not in bases:
        bases.append(config['base'])
    sources = {name: subprocess.check_output(['git','show',
               commit+':reproduction/prefix_diversity200/'+name+'.py'],cwd=repo,text=True)
               for name in ('cost_snapshot', 'cost_retrieval')}
    payload = dict(commit=commit, run_id=config['run_id'], bases=bases, sources=sources)
    script = '''import os,json,time,hashlib
from pathlib import Path
os.nice(19)
p=PAYLOAD
out={}
for name,source in p['sources'].items():
 scope={'__name__':'published_cost_reader'};exec(compile(source,name+'.py','exec'),scope)
 kwargs={'maximum_new_outers':4} if name=='cost_retrieval' else {}
 value=scope['collect'](p['bases'],p['run_id'],**kwargs)
 value.update(reader_commit=p['commit'],reader_sha256=hashlib.sha256(source.encode()).hexdigest());out[name]=value
path=Path(out['cost_retrieval']['output_directory'])/('execution-'+str(time.time_ns())+'.json')
path.write_text(json.dumps(out,allow_nan=False))
print(json.dumps(out,allow_nan=False))
'''.replace('PAYLOAD', repr(payload))
    transport = work/('cost-sync-'+str(time.time_ns())+'-transport.json')
    costs = json.loads(remote(script, transport, target=config['node']))
    cost, retrieval = costs['cost_snapshot'], costs['cost_retrieval']
    generation_outer = max(map(int, cost['complete_outer_rows']))
    if generation_outer > observation['complete_outer']:
        observation = refresh(root)
    state = read(output/'STATUS.json')
    n = state['actual_complete_outer']
    history = dict(run_id=config['run_id'], observed_unix=cost['observed_unix'],
        reader_commit=commit, reader_sha256=cost['reader_sha256'], rows={
            k:dict(outer=int(k), **r['actor_grounding_reward_summary'],
                joint_receipt_sha256=r['receipt_sha256'][r['base']+'/outers/outer-%06d/complete.json'%int(k)])
            for k,r in cost['complete_outer_rows'].items()},
        scope='original saved selected512G5 actor grounding reward; distinct from EM/prefix probe and heldout QA; no new forward')
    actual = read(work/'LATEST_ACTUAL.json')
    previous_reservation = read(work/'RESERVATION_COST_EVIDENCE.json')
    job_ids = list(dict.fromkeys(['3416','3417','3418','3421'] +
        list(previous_reservation.get('jobs',{})) + [str(state['job_id'])]))
    reservation = reservation_cost(work, config['run_id'], job_ids)
    reservation.update(reader_commit=commit,
        reader_sha256=hashlib.sha256((repo/'reproduction/prefix_diversity200/cost_reservation.py').read_bytes()).hexdigest())
    for name,value in [('COST_SNAPSHOT.json',cost),('RETRIEVAL_COST_SNAPSHOT.json',retrieval),
                       ('TRAINING_REWARD_HISTORY.json',history),('RESERVATION_COST_EVIDENCE.json',reservation)]:
        save(work/name,value);save(output/name,value)
    reward_outer = max(map(int,history['rows']))
    reward = history['rows'][str(reward_outer)]
    state.update(cost_generation_complete_outer_rows=sorted(map(int,cost['complete_outer_rows'])),
        cost_generation_sampled_tokens=cost['total_completed_outer_sampled_tokens'],
        retrieval_cost_verified_complete_outers=sorted(map(int,retrieval['complete_outer_rows'])),
        retrieval_cost_verified_logical_search_actions=retrieval['total_distinct_logical_retrieval_calls'],
        retrieval_cost_pending_complete_outers=retrieval['pending_complete_outers'],
        historical_terminal_jobs_reserved_gpu_hours=reservation['terminal_jobs_gpu_hours'],
        observed_total_reserved_gpu_hours=reservation['observed_total_reserved_gpu_hours'],
        latest_actor_grounding_soft_reward_outer=reward_outer,
        latest_actor_grounding_soft_reward_mean=reward['score_mean'],
        actor_grounding_soft_reward_verified_outers=sorted(map(int,history['rows'])),
        cost_accounting_complete=False)
    save(output/'STATUS.json',state);save(root/'outputs/STATUS.json',state)
    report=(output/'REPORT.md').read_text()+('\n## Original reward and scoped costs\n\n'
        'Original saved actor grounding reward outer%d selected512G5/2560 mean%.8f, min%.8f, max%.8f. All8rank statistics finite; prefix correctness head not used as reward. Original joint receipt hashes bind history1–%d; no new forward. Selected training reward is distinct from heldout QA or proof of selection benefit.\n\n'
        'Committed generation/retrieval through%d/%d: %d sampled tokens, %d logical searches; backlog%s. Inherited prefix searches not doubled; actor512 subset not added. Failed attempts, QA, HTTP retries/cache misses and physical FAISS calls separately scoped.\n\n'
        'Historical terminal reservations %.6f GPU-hours include useful saved updates and failure/setup/QA overhead. With current job, observed reserved total%.6f GPU-hours is a lower bound for running jobs at native observation. Do not add phase wall again. Reserved CPU capacity is not actual CPU use; sacct disabled/sstat invalid, CPU snapshots partial. Goal200/final51713 and final cost report remain incomplete.\n'
        %(reward_outer,reward['score_mean'],reward['score_min'],reward['score_max'],reward_outer,
          generation_outer,max(map(int,retrieval['complete_outer_rows'])),cost['total_completed_outer_sampled_tokens'],
          retrieval['total_distinct_logical_retrieval_calls'],retrieval['pending_complete_outers'],
          reservation['terminal_jobs_gpu_hours'],reservation['observed_total_reserved_gpu_hours']))
    (output/'REPORT.md').write_text(report);(root/'outputs/REPORT.md').write_text(report)
    scalar_series=read(output/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json')
    selection=scalar_series[str(n)]['joint']['actor_selection']
    direct_outers=sorted(int(k) for k,v in scalar_series.items() if v.get('independent_direct_8rank_ledger_facts') is True)
    pending_direct_outers=sorted(int(k) for k,v in scalar_series.items() if v.get('independent_direct_8rank_ledger_facts') is not True)
    ledger_scope=f'Independent direct8rank ledger rows {direct_outers}; rows without the newer direct-read flag {pending_direct_outers} retain their original joint/CPU milestone evidence and require separate small-ledger readback. All1–{n} retain20 probe loss points.'
    handoff=(work/'CURRENT_HANDOFF.md').read_text()
    handoff=re.sub(r'^# Search-G1: .*',f'# Search-G1: {n}/200 complete, original job{state["job_id"]} continues',handoff,count=1)
    handoff=re.sub(r'Actual observation .*?\n\n',
        f'Actual observation {state["latest_actual_observed_unix"]}: {state["native_state"]}, stage {state["stage"]}, {state["node"]}. Base {state["base"]}; frozen source {state["frozen_source"]}; config SHA256 {state["config_sha256"]}. Saved joint actor/probe Adam {state["joint_actor_adam_per_rank"]}/{state["joint_probe_adam_per_rank"]}; historical20/rank unsaved probe separate. Healthy training remains continuous.\n\n',handoff,count=1,flags=re.S)
    handoff=re.sub(r'Outer\d+:.*?\n\n',
        f'Outer{n}: continued mixed{selection["continued1024"]["mixed_questions"]}/1024; actor mixed{selection["actor512"]["mixed_questions"]}/512. No extra sampling; EM-same groups fill512. {ledger_scope} Recent mean whole-outer{state["recent_actual_whole_outer_mean_seconds"]}s. QA latest outer{state["latest_qa_outer"]}, EM{state["latest_qa_em"]}/F1{state["latest_qa_f1"]}, count{state["latest_qa_count"]}.\n\n',handoff,count=1,flags=re.S)
    handoff=re.sub(r'Cost coverage:.*?\n\n',
        f'Cost coverage: generation through{generation_outer}, retrieval through{max(map(int,retrieval["complete_outer_rows"]))}, {cost["total_completed_outer_sampled_tokens"]} sampled tokens/{retrieval["total_distinct_logical_retrieval_calls"]} logical searches. Reward history through{reward_outer}, mean{reward["score_mean"]}, distinct from QA. Use published python3 -m reproduction.prefix_diversity200.sync_cost_local --root {root} --publish to observe/sync/publish without training actions; bounded CPU cost readers execute outside frozen runtime. Remote proof cache preparation/retrieval-cost-v1, at most4 new complete outers per call. Prefix searches not doubled; actor512 subset not added. No raw/model output returns locally. Failed attempts/QA/physical HTTP/cache/query counts separate.\n\n',handoff,count=1,flags=re.S)
    handoff=re.sub(r'brings observed total to[0-9.]+',f'brings observed total to{reservation["observed_total_reserved_gpu_hours"]:.6f}',handoff)
    handoff=re.sub(r'samples\d+, errors\[\]',f'samples{state["resource_sampler_samples_verified"]}, errors[]',handoff)
    (work/'CURRENT_HANDOFF.md').write_text(handoff)
    names=['STATUS.json','REPORT.md','ITERATION_METRICS.csv','COMPLETED_ITERATION_SCALAR_EVIDENCE.json',
        'QA_MILESTONE_HISTORY.json','RESOURCE5S_SUMMARY_LATEST.json','COST_SNAPSHOT.json',
        'RETRIEVAL_COST_SNAPSHOT.json','RESERVATION_COST_EVIDENCE.json','TRAINING_REWARD_HISTORY.json']
    names+=sorted(p.name for p in output.glob('QA*_SUMMARY.json'))
    if (output/'FINAL_QA_SUMMARY.json').exists():names.append('FINAL_QA_SUMMARY.json')
    mirrors=[root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/search-g1-prefix1024-diversity512',
        Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/search-g1-prefix1024-diversity512')]
    for name in names:
        shutil.copyfile(output/name,evidence/name)
        for destination in mirrors:shutil.copyfile(output/name,destination/name)
    shutil.copyfile(work/'CURRENT_HANDOFF.md',evidence/'CURRENT_HANDOFF.md')
    publication=None
    if publish:
        subprocess.run(['git','diff','--check'],cwd=repo,check=True)
        paths=[str((evidence/name).relative_to(repo)) for name in names+['CURRENT_HANDOFF.md']]
        if not set(git(repo,'diff','--cached','--name-only').splitlines()).issubset(paths):
            raise ValueError('Preserve unrelated staged work; results publication not attempted')
        subprocess.run(['git','add','--',*paths],cwd=repo,check=True)
        changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode
        if changed not in (0,1):raise ValueError('Unable to inspect staged audit files')
        if changed:
            subprocess.run(['git','commit','-m',f'Preserve complete{n} actual progression and scoped evidence'],cwd=repo,check=True,stdout=subprocess.PIPE)
            subprocess.run(['git','push','origin','HEAD'],cwd=repo,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        published=git(repo,'rev-parse','HEAD')
        if git(repo,'ls-remote','origin','refs/heads/'+branch).split()[0]!=published:
            raise ValueError('Results push not confirmed; training is unchanged')
        files={}
        tracked=git(repo,'ls-files','reproduction/prefix_diversity200/control_evidence').splitlines()
        tracked+=['reproduction/prefix_diversity200/'+name+'.py' for name in
                  ('cost_snapshot','cost_reservation','cost_retrieval','refresh_local','sync_cost_local')]
        for path in tracked:
            content=subprocess.check_output(['git','show',published+':'+path],cwd=repo)
            if content!=(repo/path).read_bytes():raise ValueError('Published audit file differs')
            files[path]=hashlib.sha256(content).hexdigest()
        publication=dict(commit=published,remote_sha=published,branch=branch,
            verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
            complete_outer=n,live_job=str(state['job_id']),observed_actual_evidence=state['latest_actual_evidence'],
            frozen_training_source=state['frozen_source'],committed_generation_cost_through_outer=generation_outer,
            committed_logical_retrieval_cost_through_outer=max(map(int,retrieval['complete_outer_rows'])),
            original_actor_soft_reward_history_through_outer=reward_outer,
            full_cost_accounting_complete=False,goal_complete=False)
        save(output/'RESULTS_PUBLICATION.json',publication);state['results_publication_commit']=published
        save(output/'STATUS.json',state);save(root/'outputs/STATUS.json',state)
        for destination in mirrors:
            for name in ('STATUS.json','RESULTS_PUBLICATION.json'):shutil.copyfile(output/name,destination/name)
    log_errors=[dict(file=log['file'],line=line) for log in actual.get('log_tails',[])
        for line in log['tail'].splitlines() if any(x in line for x in
        ('Traceback (most recent call last)','RuntimeError:','FloatingPointError:'))]
    return dict(observation=observation,complete_outer=n,stage=state['stage'],native=state['native_state'],
        actor_saved=state['actual_actor_adam_per_rank'],probe_saved=state['actual_probe_adam_per_rank'],
        actor512=selection['actor512'],continued1024=selection['continued1024'],
        recent_whole_outer_seconds=state['recent_actual_whole_outer_mean_seconds'],reward_outer=reward_outer,
        reward_mean=reward['score_mean'],qa_outer=state['latest_qa_outer'],qa_em=state['latest_qa_em'],qa_f1=state['latest_qa_f1'],
        generation_cost_through=generation_outer,retrieval_cost_through=max(map(int,retrieval['complete_outer_rows'])),
        sampled_tokens=cost['total_completed_outer_sampled_tokens'],logical_searches=retrieval['total_distinct_logical_retrieval_calls'],
        retrieval_backlog=retrieval['pending_complete_outers'],recent_log_errors=log_errors,
        resource_errors=actual['resource_summary']['latest']['errors'],
        publication=publication['commit']if publication else None,training_modified=False,goal_complete=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();print(json.dumps(sync(args.root,args.publish),allow_nan=False))
