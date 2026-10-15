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
