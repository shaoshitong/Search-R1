"""Own-file read-only extractor. Executed only by the separately gated local entry.

The local entry prepends the SHA-verified published candidate aggregation
functions. This file never imports training/model code or writes remote files.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


COMMITS = ['0befbaf99fc44942105831b3fbec94e4bf1b8ccb',
           'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4',
           '778b2b43dad90fc0c1c5d95d77c2326286011b73']
CONFIGS = ['2a317628c497bba932698bf6d7bf64ca3503233d32080a389a36e1f69b362243',
           '22beaec8aa2b449b496754aacfd9b1e7c89009c9c27a8eff3aef92b768b28b59',
           '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3']
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
TOKEN = re.compile(r'([a-zA-Z0-9_./-]+):(-?[0-9]+(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?)(?=\s|$)')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'Duplicate original JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite original JSON')))


def original(ref, run, limit=100000000, prefix=False):
    path = Path(ref['path'])
    need(path.is_absolute() and '..' not in path.parts and not path.is_symlink(), 'Original path identity')
    root = Path(run).resolve(strict=True)
    need(path.resolve(strict=True).is_relative_to(root), 'Original path leaves own run')
    need(path.is_file(), 'Original ordinary file absent')
    before = path.stat()
    amount = ref.get('bytes', before.st_size)
    need(type(amount) is int and 0 < amount <= limit, 'Original scalar file/prefix size budget')
    need(before.st_size >= amount if prefix else before.st_size == amount, 'Original bytes differ')
    with path.open('rb') as stream:
        raw = stream.read(amount)
    after = path.stat()
    if prefix:
        with path.open('rb') as stream:
            second = stream.read(amount)
        need(raw == second and len(raw) == amount and raw.endswith(b'\n'), 'Original log prefix unstable/incomplete')
    else:
        need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Original file changed')
    need(len(raw) == amount, 'Short original read')
    if ref.get('sha256') is not None:
        need(sha(raw) == ref['sha256'], 'Previously pinned original SHA changed')
    if ref.get('mtime_ns') is not None and not prefix:
        need(before.st_mtime_ns == ref['mtime_ns'], 'Previously pinned original mtime changed')
    return raw, dict(path=str(path), resolved_path=str(path.resolve()), bytes=len(raw), sha256=sha(raw),
                     mtime_ns_before=before.st_mtime_ns, mtime_ns_after=after.st_mtime_ns,
                     stable_read=True, read_scope='stable exact prefix read twice' if prefix else 'complete stable original file',
                     historical_capture_sha256=ref.get('historical_capture_sha256'),
                     expected_input_sha256=ref.get('sha256'))


def metric_lines(ref, run):
    raw, source = original(ref, run, 256000000, prefix=True)
    found = {}
    offset = 0
    for ordinal, line in enumerate(raw.splitlines(keepends=True), 1):
        start = offset
        offset += len(line)
        text = line.decode('utf-8')
        if 'timing_s/step:' not in text:
            continue
        metric = {}
        for key, number in TOKEN.findall(text):
            value = float(number)
            need(math.isfinite(value) and (key not in metric or metric[key] == value), 'Conflicting original metric scalar')
            metric[key] = value
        step = metric.get('step')
        need(type(step) in (int, float) and step == int(step), 'Original complete metric has no integer step')
        step = int(step)
        item = dict(path=source['path'], resolved_path=source['resolved_path'], byte_start=start, byte_end=offset,
                    line_number=ordinal, line_sha256=sha(line), raw_line_utf8=text, prefix=source)
        if step in found:
            need(found[step]['metric'] == metric, 'Ambiguous original metric for one accepted outer')
            found[step]['identical_duplicate_lines'].append(item)
        else:
            found[step] = dict(metric=metric, source=item, identical_duplicate_lines=[])
    return found, source


def rank_record(ref, run, commit, config, outer, rank, attempt, writer_sha):
    need(Path(ref['path']).parent == Path(run)/'output/tree-rank-audit', 'Own admitted rank directory')
    raw, source = original(ref, run, 262144)
    value = strict(raw)
    match = re.fullmatch(r'step-(\d{6})(?:-attempt-(\d{6}))?-rank-(\d{2})\.json', Path(ref['path']).name)
    need(match and int(match[1]) == outer and int(match[3]) == rank, 'Admitted rank filename')
    if match[2] is not None:
        need(int(match[2]) == attempt and value.get('attempt_id') == attempt, 'Original rank attempt')
    for key, expected in dict(kind='tree_actor_rank', outer_step=outer, rank=rank, world_size=8,
                              code_commit=commit, config_sha256=config, data_identity=DATA,
                              run_id=Path(run).name, auditor_source_sha256=writer_sha).items():
        need(value.get(key) == expected, 'Original rank identity: ' + key)
    calls = value.get('successful_optimizer_calls')
    need(type(calls) is int and 1 <= calls <= 20, 'Credited rank has no actual positive writes')
    sample = value.get('local_shard_sample_delta') or {}
    need(value.get('finite_grad_norms') == calls and 1 <= value.get('nonzero_grad_norms', 0) <= calls,
         'Original rank finite/nonzero gradients')
    need(type(sample.get('count')) is int and sample['count'] > 0 and sample.get('finite') == sample['count']
         and 1 <= sample.get('nonzero', 0) <= sample['count'], 'Original finite/nonzero shard sample')
    processed = value.get('opportunities_processed')
    skipped = value.get('opportunities_skipped', value.get('skipped_optimizer_calls'))
    need(processed is None or type(processed) is int and processed == 20, 'Explicit rank opportunities')
    need(skipped is None or type(skipped) is int and 0 <= skipped <= 19 and calls + skipped == 20,
         'Explicit rank skipped opportunities')
    return dict(rank=rank, successful_optimizer_calls=calls,
                skipped_empty_minibatches=value.get('skipped_empty_minibatches'),
                opportunities_processed=processed, opportunities_skipped=skipped,
                finite_grad_norms=value['finite_grad_norms'], nonzero_grad_norms=value['nonzero_grad_norms'],
                local_shard_sample_delta=sample, source=source)


def physical_attempt(ref, run, commit, config, outer, attempt, expected_ranks):
    need(Path(ref['path']).name == f'attempt-{attempt:06d}.json' and
         Path(ref['path']).parent.name == 'attempt-results' and
         Path(ref['path']).parent.parent.name == 'training_state', 'Own original attempt-result path')
    raw, source = original(ref, run, 1000000)
    record = strict(raw)
    need(record.get('protocol') == 'tree_fresh_actor_result_v1' and
         record.get('status') == 'COMPLETED_OPPORTUNITIES_OUTER_CREDITED' and
         record.get('accepted_outer') == outer and record.get('attempt_id') == attempt, 'Original credited attempt')
    result = record['all_rank_result']
    need(result.get('decision') == 'COMPLETED_OUTER' and result.get('accepted_outer') == outer and
         result.get('attempt_id') == attempt and result.get('completed_outer_before') == outer - 1,
         'Original all-rank gathered result')
    identity = result['identity']
    for key, expected in dict(code_commit=commit, config_sha256=config, model_identity=MODEL,
                              data_identity=DATA, run_id=Path(run).name).items():
        need(identity.get(key) == expected, 'Original attempt identity: ' + key)
    values = result['ranks']
    need(len(values) == 8 and [value.get('rank') for value in values] == list(range(8)), 'Original gathered ranks')
    physical = []
    for rank, value in enumerate(values):
        calls = expected_ranks[rank]['successful_optimizer_calls']
        ledger = value.get('opportunity_ledger')
        validate_ledger(ledger, calls, allow_zero=False)
        need(type(value.get('optimizer_calls')) is int and value['optimizer_calls'] == calls and
             value.get('opportunities_processed') == 20 and value.get('opportunities_skipped') == 20 - calls,
             'Actual gathered opportunity counters')
        need(isinstance(ledger, list) and len(ledger) == 20 and
             all(type(row.get('actual_writes')) is int and row['actual_writes'] in (0, 1) for row in ledger) and
             sum(row['actual_writes'] for row in ledger) == calls, 'Actual physical opportunity ledger')
        direct = value['direct_rank_record']
        path = Path(run) / 'output/fresh-state' / f'outer-{outer:06d}-attempt-{attempt:06d}-rank{rank:02d}.json'
        need(Path(direct['path']).resolve() == path.resolve(), 'Physical rank path identity')
        blob, meta = original(dict(path=str(path), sha256=direct['sha256']), run, 1000000)
        state = strict(blob)
        expected = dict(rank=rank, world_size=8, attempt_id=attempt, observed_outer=outer,
                        actual_optimizer_calls=calls, opportunities_processed=20,
                        opportunities_skipped=20-calls, optimizer_opportunity_ledger=ledger,
                        scheduler_step_before=outer-1, scheduler_step_after=outer)
        expected.update({key: identity[key] for key in ('job_id', 'run_id', 'code_commit', 'config_sha256', 'data_identity', 'model_identity')})
        need(all(state.get(key) == item for key, item in expected.items()), 'Physical rank bytes disagree with gathered result')
        physical.append(dict(rank=rank, source=meta, opportunities_processed=20, opportunities_skipped=20-calls,
                             opportunity_ledger_sha256=sha(json.dumps(ledger, sort_keys=True, separators=(',', ':')).encode())))
    return dict(source=source, physical_ranks=physical,
                actual_completed_optimizer_calls_per_rank=record['actual_completed_optimizer_calls_per_rank'])


def candidates(value):
    rows, rounds = value.get('candidate_attempts'), value.get('candidate_rounds')
    need(isinstance(rows, list) and isinstance(rounds, list), 'Original candidate/round rows absent')
    unique, seen, duplicate, missing = [], set(), 0, 0
    for row in rounds:
        identity = row.get('round_id')
        if identity is None:
            missing += 1
        elif identity in seen:
            duplicate += 1
            continue
        else:
            seen.add(identity)
        unique.append(row)
    sums, known, counts = ledger(unique)
    stages = {}
    for stage in sorted({row.get('stage', 'UNKNOWN') for row in rows + unique}):
        selected = [row for row in unique if row.get('stage') == stage]
        ss, sk, sc = ledger(selected)
        stages[stage] = dict(candidate_outcomes=grouping([row for row in rows if row.get('stage') == stage]),
                             round_controller_ledger=ss, round_ledger_known_sums=sk,
                             round_ledger_field_record_counts=sc, unique_round_count=len(selected))
    return dict(candidate_outcomes=grouping(rows), stages=stages, candidate_round_controller_ledger=sums,
                candidate_round_ledger_known_sums=known, candidate_round_ledger_field_record_counts=counts,
                duplicate_round_ids_excluded=duplicate, rounds_missing_ids=missing,
                unique_round_count=len(unique),
                generation_candidate_round_elapsed_seconds=elapsed(unique, 'generation_wall_seconds') if not missing else None,
                scoring_candidate_round_elapsed_seconds=elapsed(unique, 'scoring_wall_seconds') if not missing else None,
                padding_cost_by_candidate_outcome=None, independent_outcome_wall_seconds=None,
                question_count=len(value['question_groups']), training_leaf_count=len(value['leaves']))


def native_inventory(run, commit, outers):
    available=[];archive=Path(run)/'output/native-manifest-history'
    if archive.exists():
        for p in archive.glob('source-'+commit+'-outer-*.manifest.json'):
            match=re.fullmatch('source-'+commit+r'-outer-(\d{6})\.manifest\.json',p.name)
            if match and int(match[1]) in outers:available.append((p,int(match[1])))
    states=Path(run)/'checkpoints/training_state'
    for prefix in (states,states/'rolling-completed'):
        if prefix.exists():
            for p in prefix.glob('global_step_*/manifest.json'):
                match=re.fullmatch(r'global_step_(\d+)',p.parent.name)
                if match and int(match[1]) in outers:available.append((p,int(match[1])))
    return sorted(available,key=lambda item:(item[1],str(item[0])))


def prepare(seed):
    """One separate own-file discovery call. No audit payload or PT returned."""
    start,cpu=time.time(),time.process_time();os.nice(19)
    need(seed['schema']=='tree_final200_original_prepare_seed_v1','Prepare seed schema')
    result=dict(schema='tree_final200_original_remote_pins_v1',runs=[],periodic_controls=[])
    for index,commit in enumerate(COMMITS):
        run='/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:'+commit
        outers=[1] if index==0 else list(range(2,6)) if index==1 else list(range(6,201))
        original(dict(path=run+'/resolved.yaml',sha256=CONFIGS[index]),run,1000000)
        stage=Path(run)/'stage.log';need(stage.is_file() and stage.stat().st_size<=256000000,'Own original log budget')
        found={};offset=0;digest=hashlib.sha256();prefix_end=0;prefix_sha=None
        with stage.open('rb') as stream:
            for raw in stream:
                offset+=len(raw);digest.update(raw)
                if b'timing_s/step:' not in raw:continue
                metric={k:float(v) for k,v in TOKEN.findall(raw.decode('utf-8'))}
                step=metric.get('step')
                if step not in outers:continue
                need(raw.endswith(b'\n') and (step not in found or found[step]==metric),'Ambiguous/incomplete original metric')
                found[step]=metric;prefix_end=offset;prefix_sha=digest.copy().hexdigest()
        need(set(found)==set(outers),'Prepare requires complete original accepted metrics1..200')
        stage_ref=dict(path=str(stage),bytes=prefix_end,sha256=prefix_sha)
        original(stage_ref,run,256000000,prefix=True)
        attempts={};attempt_refs={};noncredited=[]
        directory=Path(run)/'checkpoints/training_state/attempt-results'
        if index==2:
            need(directory.is_dir(),'Own child credited attempt directory')
            for p in sorted(directory.glob('attempt-*.json')):
                if not re.fullmatch(r'attempt-\d{6}\.json',p.name):continue
                raw,meta=original(dict(path=str(p)),run,1000000);value=strict(raw)
                n=value.get('accepted_outer')
                if value.get('status')=='COMPLETED_OPPORTUNITIES_OUTER_CREDITED' and n in outers:
                    need(n not in attempts,'Ambiguous credited attempt for one outer')
                    attempts[n]=value['attempt_id'];attempt_refs[n]=dict(path=str(p))
                else:noncredited.append(dict(path=str(p)))
            need(set(attempts)==set(outers),'Prepare child accepted attempt coverage')
        controllers={};rankmap={}
        for p in sorted((Path(run)/'output/training-rollout-cost').glob('attempt-*.json')):
            if not re.fullmatch(r'attempt-\d{6}\.json',p.name):continue
            raw,meta=original(dict(path=str(p)),run,262144);value=strict(raw);n=value.get('candidate_outer')
            if (n in outers and value.get('status')=='TRAINING_ROLLOUT_COST_COMPLETE' and
                    (index!=2 or value.get('attempt_id')==attempts[n])):
                need(n not in controllers,'Ambiguous original Controller')
                controllers[n]=(p,value)
                if index!=2:attempts[n]=value.get('attempt_id')
            else:noncredited.append(dict(path=str(p)))
        for p in sorted((Path(run)/'output/tree-rank-audit').glob('*.json')):
            match=re.fullmatch(r'step-(\d{6})(?:-attempt-(\d{6}))?-rank-(\d{2})\.json',p.name)
            if not match:continue
            raw,meta=original(dict(path=str(p)),run,262144);value=strict(raw)
            need(value.get('kind')=='tree_actor_rank' and value.get('code_commit')==commit,'Own admitted rank identity')
            n,rank=int(match[1]),int(match[3]);calls=value.get('successful_optimizer_calls')
            selected=n in outers and type(calls) is int and calls>0 and (index!=2 or value.get('attempt_id')==attempts[n])
            if selected:
                need((n,rank) not in rankmap,'Ambiguous positive-write original rank')
                rankmap[(n,rank)]=dict(path=str(p))
            else:noncredited.append(dict(path=str(p)))
        rows=[]
        for n in outers:
            need(n in controllers and all((n,r) in rankmap for r in range(8)),'Prepare Controller/all-rank coverage')
            captured=seed['captured_audits'].get(str(n),seed['captured_audits'].get(n))
            if captured:
                audit=Path(run)/'output/tree-audit'/captured['audit_file']
                need(audit.is_file() and audit.stat().st_size==captured['bytes'],'Preserved audit stat')
                audit_ref=dict(path=str(audit),bytes=captured['bytes'],sha256=captured['sha256'])
            else:
                options=[p for p in (Path(run)/'output/tree-audit').glob(f'step_{n}_rollout_*.json')
                         if re.fullmatch(r'step_'+str(n)+r'_rollout_\d+\.json',p.name)]
                if len(options)>1:
                    selected=[]
                    for p in options:
                        header={}
                        with p.open('rb') as stream:
                            for line in stream:
                                if stream.tell()>262144:break
                                match=re.match(rb'  "(attempt_id|global_step)":\s*([^,]+),?',line)
                                if match:
                                    header[match[1].decode()]=json.loads(match[2].rstrip(b',\r\n'))
                                if len(header)==2:break
                        if header.get('global_step')==n and header.get('attempt_id')==attempts.get(n):selected.append(p)
                    options=selected
                need(len(options)==1,'Ambiguous/missing audit; no newest-file guess')
                audit_ref=dict(path=str(options[0]),bytes=options[0].stat().st_size,historical_capture_sha256=None)
            row=dict(outer=n,audit=audit_ref,controller=dict(path=str(controllers[n][0])),
                     ranks=[rankmap[(n,r)] for r in range(8)])
            if type(attempts.get(n)) is int:row['attempt_id']=attempts[n]
            if index==2:row['attempt_result']=attempt_refs[n]
            rows.append(row)
        if index==2:
            accepted=set(attempts.values())
            for p in sorted((Path(run)/'output/fresh-state').glob('outer-*-attempt-*-rank*.json')):
                match=re.fullmatch(r'outer-\d{6}-attempt-(\d{6})-rank\d{2}\.json',p.name)
                if match and int(match[1]) not in accepted:noncredited.append(dict(path=str(p)))
        terminal=Path(run)/'run-result.json'
        if terminal.is_file():noncredited.append(dict(path=str(terminal)))
        native=[dict(path=str(p),outer=n,bytes=p.stat().st_size) for p,n in native_inventory(run,commit,outers)]
        result['runs'].append(dict(source_commit=commit,config_sha256=CONFIGS[index],run=run,
                                  rank_writer_sha256=seed['rank_writer_sha256'],stage=stage_ref,rows=rows,
                                  native_metadata=native,noncredited_records=noncredited))
    run=result['runs'][-1]['run'];controls={}
    for p in sorted((Path(run)/'output/validation-cost').glob('*.json')):
        if not re.fullmatch(r'validation-\d+-\d+\.json',p.name):continue
        if p.name in seed['old_control_names']:continue
        raw,meta=original(dict(path=str(p)),run,262144);value=strict(raw);n=value.get('actual_completed_outer')
        if n in (195,200) and value.get('status')=='VALIDATION_COST_COMPLETE' and value.get('phase')=='periodic':
            need(n not in controls,'Ambiguous new original dev control')
            controls[n]=dict(outer=n,path=str(p),bytes=len(raw),sha256=sha(raw))
    need(set(controls)=={195,200},'Prepare exact completed195+200 control sources')
    result['periodic_controls']=[controls[n] for n in (195,200)]
    return dict(schema='tree_final200_original_prepare_readback_v1',status='ORIGINAL200_PATHS_PREPARED_NOT_COST_ACCEPTANCE',
                remote_pins=result,remote_calls_this_preparation=1,collector_remote_calls_this_preparation=0,
                observer_wall_seconds=time.time()-start,observer_process_cpu_seconds=time.process_time()-cpu,
                remote_files_written=False,raw_audit_or_asset_returned=False)


def collect(pins):
    started, cpu = time.time(), time.process_time()
    os.nice(19)
    need(pins['schema'] == 'tree_final200_original_remote_pins_v1', 'Original input schema')
    need([r['source_commit'] for r in pins['runs']] == COMMITS, 'Own lineage run order')
    rows, stage_sources, native_records, extra, native_missing = [], [], [], [], []
    for index, runpin in enumerate(pins['runs']):
        commit, config, run = COMMITS[index], CONFIGS[index], runpin['run']
        expected_run = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + commit
        need(run == expected_run and runpin['config_sha256'] == config, 'Own source/config/run identity')
        config_raw, config_source = original(dict(path=run+'/resolved.yaml', sha256=config), run, 1000000)
        metrics, stage_source = metric_lines(runpin['stage'], run)
        stage_sources.append(stage_source)
        outers = [1] if index == 0 else list(range(2, 6)) if index == 1 else list(range(6, 201))
        need([row['outer'] for row in runpin['rows']] == outers, 'Exact effective original outer inventory')
        for row in runpin['rows']:
            outer = row['outer']
            need(outer in metrics, 'Original complete metric missing: '+str(outer))
            m = metrics[outer]
            need(Path(row['audit']['path']).parent == Path(run)/'output/tree-audit' and
                 re.fullmatch(r'step_'+str(outer)+r'_rollout_\d+\.json', Path(row['audit']['path']).name),
                 'Own original audit directory/filename')
            audit_raw, audit_source = original(row['audit'], run)
            audit = strict(audit_raw)
            need(audit.get('status') == 'ROLLOUT_DIAGNOSTIC_COMPLETE' and audit.get('global_step') == outer,
                 'Original accepted audit status/outer')
            need(Path(row['controller']['path']).parent == Path(run)/'output/training-rollout-cost' and
                 re.fullmatch(r'attempt-\d{6}\.json', Path(row['controller']['path']).name), 'Own Controller path')
            controller_raw, controller_source = original(row['controller'], run, 262144)
            controller = strict(controller_raw)
            for key, expected in dict(status='TRAINING_ROLLOUT_COST_COMPLETE', candidate_outer=outer,
                                      code_commit=commit, config_sha256=config, model_identity=MODEL, data_identity=DATA).items():
                need(controller.get(key) == expected, 'Original Controller identity: '+key)
            attempt = audit.get('attempt_id')
            if attempt is None:
                need(index != 2, 'Child original audit must retain its actual attempt')
                attempt = controller.get('attempt_id', row.get('attempt_id'))
            need(type(attempt) is int and attempt > 0, 'Original accepted attempt identity missing')
            need(row.get('attempt_id') is None or row['attempt_id'] == attempt, 'Pinned original attempt differs')
            need(controller.get('attempt_id', attempt) == attempt, 'Original Controller attempt')
            need(len(row['ranks']) == 8, 'Exact eight original rank file refs required')
            ranks = [rank_record(ref, run, commit, config, outer, rank, attempt, runpin['rank_writer_sha256'])
                     for rank, ref in enumerate(row['ranks'])]
            need(m['metric'].get('actor/optimizer_steps') == sum(r['successful_optimizer_calls'] for r in ranks)/8,
                 'Whole metric and physical all-rank writes disagree')
            attempt_record = None
            if index == 2:
                need(row.get('attempt_result') is not None, 'Child original attempt-result ref required')
                attempt_record = physical_attempt(row['attempt_result'], run, commit, config, outer, attempt, ranks)
            elif row.get('attempt_result') is not None:
                need(False, 'Do not apply the child fresh-guard schema to inherited sources')
            rows.append(dict(outer=outer, source_commit=commit, config_sha256=config, run=run, attempt_id=attempt,
                             evidence_mode='RETROSPECTIVE_SOURCE_READBACK', metric=m['metric'], metric_line=m['source'],
                             identical_duplicate_metric_lines=m['identical_duplicate_lines'], controller=controller,
                             controller_source=controller_source, config_source=config_source, audit_source=audit_source,
                             audit_file=Path(row['audit']['path']).name, audit_file_bytes=len(audit_raw),
                             audit_file_sha256=audit_source['sha256'], global_step=outer, rank_records=ranks,
                             attempt_record=attempt_record, branch=candidates(audit)))
        available=[]
        archive=Path(run)/'output/native-manifest-history'
        if archive.exists():
            for p in archive.glob('source-'+commit+'-outer-*.manifest.json'):
                match=re.fullmatch('source-'+commit+r'-outer-(\d{6})\.manifest\.json',p.name)
                if match and int(match[1]) in outers:available.append(p)
        states=Path(run)/'checkpoints/training_state'
        for prefix in (states,states/'rolling-completed'):
            if prefix.exists():
                for p in prefix.glob('global_step_*/manifest.json'):
                    match=re.fullmatch(r'global_step_(\d+)',p.parent.name)
                    if match and int(match[1]) in outers:available.append(p)
        need({str(p) for p in available}=={ref['path'] for ref in runpin['native_metadata']},
             'Freeze every available own native JSON metadata path; no PT traversal')
        seen_native=set()
        for ref in runpin['native_metadata']:
            need(ref['outer'] in outers,'Historical native outer outside effective lineage')
            path = Path(ref['path'])
            need((path.parent == Path(run)/'output/native-manifest-history' and path.name.endswith('.manifest.json')) or
                 (path.name == 'manifest.json' and 'training_state' in path.parts), 'Own native JSON metadata path')
            raw, meta = original(ref, run, 1000000)
            native = strict(raw);identity = native['identity']
            for key, expected in dict(code_commit=commit, config_sha256=config, model_identity=MODEL,
                                      data_identity=DATA, completed_step=ref['outer']).items():
                need(identity.get(key) == expected, 'Historical native identity: '+key)
            need(native.get('status') in ('COMPLETE_NATIVE_ROLLING_STATE_SAVED', 'COMPLETE_SAME_WORLD_STATE_SAVED') and
                 native['driver']['next_step'] == ref['outer']+1, 'Complete original native metadata')
            if index == 2:
                guard = identity['guard_continue_accounting'];selected = rows[ref['outer']-1]['attempt_record']
                need(guard['completed_outer_updates'] == ref['outer'] and
                     guard['completed_optimizer_calls_per_rank'] == selected['actual_completed_optimizer_calls_per_rank'],
                     'Historical native versus original attempt credited counters')
            native_records.append(dict(outer=ref['outer'], source_commit=commit, source=meta, value=native))
            seen_native.add(ref['outer'])
        native_missing.append(dict(source_commit=commit,unavailable_native_JSON_outers=sorted(set(outers)-seen_native),
                                   scope='Available retained JSON inventory only; missing historical snapshot is not fabricated and no PT availability is inferred.'))
        for ref in runpin['noncredited_records']:
            path = Path(ref['path'])
            need(path.suffix == '.json' and (path.parent.name in ('attempt-results', 'fresh-state', 'tree-rank-audit',
                 'training-rollout-cost') or path == Path(run)/'run-result.json'), 'Own noncredited JSON record path')
            raw, meta = original(ref, run, 1000000)
            extra.append(dict(source_commit=commit, source=meta, value=strict(raw), effective_outer_credit=0))
    need([row['outer'] for row in rows] == list(range(1, 201)), 'Original effective200 coverage')
    controls = []
    need([p['outer'] for p in pins['periodic_controls']] == [195,200], 'Only actual195+200 newly read controls')
    for ref in pins['periodic_controls']:
        outer = ref['outer']; index = 1 if outer == 5 else 2; run = pins['runs'][index]['run']
        need(Path(ref['path']).parent == Path(run)/'output/validation-cost', 'Original periodic control path')
        raw, meta = original(ref, run, 262144);value = strict(raw)
        for key, expected in dict(code_commit=COMMITS[index], config_sha256=CONFIGS[index],
                                  model_identity=MODEL, data_identity=DATA, actual_completed_outer=outer,
                                  status='VALIDATION_COST_COMPLETE', phase='periodic', evaluated_rows=512, dataset_rows=512).items():
            need(value.get(key) == expected, 'Original periodic control identity: '+key)
        metric = value['metrics']
        need(metric.get('val/qa_count/all') == 512 and metric.get('val/qa_count/nq') == metric.get('val/qa_count/hotpotqa') == 256,
             'Original dev512 split coverage')
        need(re.fullmatch('[0-9a-f]{64}', value.get('ledger_sha256', '')) is not None, 'Original inner ledger digest')
        full = rows[outer-1]['metric']
        for key in ('val/f1/all', 'val/em/all', 'val/qa_count/all', 'val/qa_count/nq', 'val/qa_count/hotpotqa'):
            need(metric.get(key) == full.get(key), 'Original dev control and step metric disagree')
        controls.append(dict(outer=outer, source_commit=COMMITS[index], source=meta, raw_summary_utf8=raw.decode(), value=value))
    result = dict(schema='tree_final200_original_remote_readback_v1', status='ORIGINAL200_RECORDS_READ',
                  observed_epoch=time.time(), started_epoch=started, rows=rows, controls=controls,
                  stage_sources=stage_sources, native_metadata=native_records,native_metadata_missing=native_missing,
                  noncredited_records=extra,
                  observer_process_cpu_seconds=time.process_time()-cpu, observer_wall_seconds=time.time()-started,
                  remote_nice=os.getpriority(os.PRIO_PROCESS, 0), remote_files_written=False,
                  tensor_model_or_raw_rollout_transferred=False, historical_live_poll_or_FULL_created=False)
    need(result['remote_nice'] == 19, 'Own observer priority')
    return result
