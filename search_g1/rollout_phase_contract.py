"""Explicit execution limits for the independent trajectory rollout phase."""
DEFAULT_ROLLOUT_PHASE = dict(
    mode='independent_trajectory_phase_v1', agent_workers=256,
    max_inflight_trajectories=256, generation_max_num_seqs=64,
    generation_max_batched_tokens=8192, retrieval_concurrency=16,
    tool_timeout=30.0, tool_retries=2, engine_step_budget=8,
)


def validate(value, world_size=4):
    if not isinstance(value, dict) or set(value) != set(DEFAULT_ROLLOUT_PHASE):
        raise ValueError('Incomplete rollout phase contract')
    if value['mode'] != DEFAULT_ROLLOUT_PHASE['mode']:
        raise ValueError('Unsupported independent trajectory rollout mode')
    if type(world_size) is not int or world_size not in (4, 8):
        raise ValueError('This rollout phase requires four or eight independent TP1 engines')
    for key in ('agent_workers', 'max_inflight_trajectories', 'generation_max_num_seqs',
                'generation_max_batched_tokens', 'retrieval_concurrency', 'engine_step_budget'):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('Invalid rollout phase limit: '+key)
    if value['max_inflight_trajectories'] > value['agent_workers']:
        raise ValueError('Trajectory slots exceed agent worker bound')
    if value['max_inflight_trajectories'] > world_size*value['generation_max_num_seqs']:
        raise ValueError('Trajectory slots exceed the actual engine sequence capacity')
    if value['generation_max_batched_tokens'] < 4096:
        raise ValueError('Engine token budget cannot fit the preserved context budget')
    if type(value['tool_timeout']) not in (int, float) or not 0 < value['tool_timeout'] <= 300:
        raise ValueError('Invalid tool timeout')
    if type(value['tool_retries']) is not int or not 0 <= value['tool_retries'] <= 5:
        raise ValueError('Invalid bounded tool retries')
    if value['engine_step_budget'] > 16:
        raise ValueError('Engine step budget would defer independent tool completions too long')
    return dict(value)


def hydra_arguments(run):
    value=run.get('infra_rollout')
    if value is None:
        return []
    value=validate(value,run['training_world_size'])
    args=['+actor_rollout_ref.rollout.async_phase_enabled=true']
    for key,number in value.items():
        if key != 'mode':
            args.append('+actor_rollout_ref.rollout.'+key+'='+str(number))
    return args


def configure_generation(config, rollout):
    """Copy explicit limits into the controller, retaining legacy default routing."""
    if not rollout.get('async_phase_enabled', False):
        return config
    value={key:rollout[key] for key in DEFAULT_ROLLOUT_PHASE if key != 'mode'}
    value['mode']=DEFAULT_ROLLOUT_PHASE['mode']
    validate(value,config.num_gpus)
    config.async_phase_enabled=True
    for key,number in value.items():
        if key != 'mode':
            setattr(config,key,number)
    return config


def scalar_metrics(output):
    """Report actual phase timings and engine work from ordinary generated output."""
    phase=output.meta_info.get('async_rollout')
    if phase is None:
        return {}
    result={}
    for key in ('completed_trajectories','generation_requests','environment_actions',
                'peak_inflight_trajectories','retrieval_concurrency','phase_elapsed_seconds'):
        value=phase[key]
        if type(value) not in (int,float):
            raise ValueError('Invalid rollout scalar: '+key)
        result['infra/rollout/'+key]=value
    if phase.get('role')=='prefix20_pause_select_continue':
        phases=phase['phases']
        if len(phases)!=2 or [row['phase'] for row in phases]!=['candidate_prefix','selected_continuation']:
            raise ValueError('Prefix20 requires two actual closed sampling phases')
        if len({row['phase_id'] for row in phases})!=2:
            raise ValueError('Prefix20 sampling phase IDs must differ')
        ranks=[]
        for event in phases:
            receipts=event['phase_metrics']
            if not isinstance(receipts,list) or len(receipts)!=4:
                raise ValueError('Prefix20 actual four-rank phase receipts missing')
            if any(type(row.get('weight_syncs')) is not int or row['weight_syncs']!=1 for row in receipts):
                raise ValueError('Prefix20 phase did not synchronize once per rank')
            ranks.extend(receipts)
        result['infra/rollout/weight_syncs_per_rank']=2
        for key in ('engine_steps','completed_requests'):
            if all(type(row.get(key)) is int for row in ranks):
                result['infra/rollout/'+key+'_sum']=sum(row[key] for row in ranks)
        return result
    if phase.get('role')=='fixed_candidate_pool_full_rollouts':
        phases=phase['phases']
        if not phases or len({p['phase_id'] for p in phases})!=len(phases):
            raise ValueError('Fixed pool requires distinct completed native sampling phases')
        receipts=[r for p in phases for r in p['phase_metrics']]
        sync_total = 0
        for index, event in enumerate(phases):
            previous = phases[index - 1] if index else {}
            reuse = event.get('snapshot_weight_reuse', False)
            if type(reuse) is not bool:
                raise ValueError('Invalid fixed-pool snapshot reuse receipt')
            retained = reuse and previous.get('retain_snapshot_weights') is True
            expected = 0 if (retained and previous.get('policy_snapshot_id') == event.get('policy_snapshot_id')) else 1
            if reuse and (type(event.get('retain_snapshot_weights')) is not bool or
                          not event.get('policy_snapshot_id')):
                raise ValueError('Incomplete fixed-pool snapshot reuse boundary')
            if any(type(r.get('weight_syncs')) is not int or r['weight_syncs'] != expected
                   for r in event['phase_metrics']):
                raise ValueError('Candidate phase weight sync differs from its actual snapshot boundary')
            sync_total += expected
        if phases[-1].get('retain_snapshot_weights', False):
            raise ValueError('Final candidate phase did not release retained sampling weights')
        result['infra/rollout/weight_syncs_per_rank']=sync_total
        result['infra/rollout/selected_trajectories']=phase['selected_trajectories']
        for key in ('engine_steps','completed_requests'):
            if all(type(r.get(key)) is int for r in receipts):
                result['infra/rollout/'+key+'_sum']=sum(r[key] for r in receipts)
        return result
    if phase.get('role')=='dynrpo7b_pause_select_continue':
        # Prefix selection and each continuation/backstop batch close their own
        # real four-rank phase; there is no synthetic top-level rank receipt.
        phases=phase['phases']
        if (not isinstance(phases,list) or len(phases)<2 or
                phases[0]['phase']!='candidate_prefix' or
                any(event['phase']!='selected_continuation' for event in phases[1:]) or
                len({event['phase_id'] for event in phases})!=len(phases)):
            raise ValueError('DynRPO requires distinct closed prefix and continuation phases')
        ranks=[]
        syncs_per_rank=[0]*4
        snapshot_ids=set()
        for event in phases:
            receipts=event['phase_metrics']
            if not isinstance(receipts,list) or len(receipts)!=4:
                raise ValueError('DynRPO actual four-rank phase receipts missing')
            for rank,row in enumerate(receipts):
                if (row.get('phase_id')!=event['phase_id'] or row.get('aborted') is not False or
                        not row.get('policy_snapshot_id') or
                        type(row.get('weight_syncs')) is not int or row['weight_syncs']!=1):
                    raise ValueError('DynRPO phase-close receipt differs from its sampling boundary')
                syncs_per_rank[rank]+=row['weight_syncs']
                snapshot_ids.add(row['policy_snapshot_id'])
            ranks.extend(receipts)
        if len(snapshot_ids)!=1 or len(set(syncs_per_rank))!=1:
            raise ValueError('DynRPO sampling snapshot or rank synchronization diverged')
        result['infra/rollout/weight_syncs_per_rank']=syncs_per_rank[0]
        for key in ('engine_steps','completed_requests'):
            if all(type(row.get(key)) is int for row in ranks):
                result['infra/rollout/'+key+'_sum']=sum(row[key] for row in ranks)
        return result
    ranks=phase['phase_metrics']
    if not isinstance(ranks,list) or not ranks:
        raise ValueError('Missing actual worker phase-close receipts')
    syncs=[row['weight_syncs'] for row in ranks]
    if any(type(value) is not int or value!=1 for value in syncs):
        raise ValueError('Sampling phase did not synchronize exactly once per rank')
    result['infra/rollout/weight_syncs_per_rank']=1
    for key in ('engine_steps','completed_requests'):
        if all(type(row.get(key)) is int for row in ranks):
            result['infra/rollout/'+key+'_sum']=sum(row[key] for row in ranks)
    return result
