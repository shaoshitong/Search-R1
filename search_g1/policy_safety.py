"""Unreduced Dual-Clip and collective old-policy monitoring with caller-owned rollback."""
import math
import torch
import torch.distributed as dist

DISCOVERY_ROLE = 'policy_safety_calibration_discovery'
CONFIRMATION_ROLE = 'policy_safety_calibration_confirmation'
INLINE_RECOVERABLE_MODE = 'inline_first_five'


class RecoverableGuard(RuntimeError):
    """A collectively decided policy/numerical guard, before the next Adam write.

    Catching this type retains all earlier actual Adam writes. Configuration,
    identity, transport and storage failures never use it. A post-write numerical
    rejection requires an active caller-owned optimizer transaction.
    """
    def __init__(self, *, stage, reasons, diagnostic=None):
        self.stage = stage
        self.reasons = list(reasons)
        self.diagnostic = diagnostic or {}
        super().__init__('Recoverable old-rollout guard at '+stage+': '+repr(self.reasons))


def recoverable_mode(actor_config, *, group=None):
    """Bind the new mode collectively; old production/calibration remains fatal."""
    import json, os
    from pathlib import Path
    errors = []; document = {}
    try:
        safety = actor_config.get('policy_safety', None)
        configured = safety.get('mode', None) if safety is not None else None
        if os.environ.get('SEARCH_G1_PROFILE') == 'common8':
            run = json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
            mode = run.get('policy_safety_mode')
            enabled = (mode == INLINE_RECOVERABLE_MODE)
            if enabled and (configured != INLINE_RECOVERABLE_MODE or safety.get('enabled') is not True or
                            run.get('policy_calibration_role', 'production') != 'production' or
                            run.get('contract_kind') != 'fresh200_warmup57'):
                raise ValueError('Recoverable guards require the explicit bound formal production mode')
            if not enabled and configured == INLINE_RECOVERABLE_MODE:
                raise ValueError('Actor recoverable mode is not bound to the run')
        else:
            enabled = False
            if configured == INLINE_RECOVERABLE_MODE:
                raise ValueError('Recoverable formal mode requires common8')
        document = dict(recoverable_guard_mode=enabled, actor_mode=configured)
    except Exception as failure:
        errors.append(type(failure).__name__+': '+str(failure))
    _phase_agreement(document, errors, group)
    return document['recoverable_guard_mode']


def agree_attempt_result(result, *, group=None):
    """All live ranks finish with the same disposition and actual write counts."""
    if dist.is_initialized():
        results = [None] * dist.get_world_size(group)
        dist.all_gather_object(results, result, group=group)
        shared = ('status', 'logical_update', 'attempt_id', 'attempt_index',
                  'optimizer_writes_before', 'optimizer_writes_after',
                  'optimizer_writes_this_attempt', 'opportunities_processed',
                  'opportunity_statuses', 'guard_stage', 'guard_reasons',
                  'complete_outer', 'full_budget_complete', 'recovery_protocol_version',
                  'optimizer_calls_this_attempt', 'optimizer_rollbacks_this_attempt',
                  'no_automatic_rollback', 'current_step_rollback',
                  'actual_monitor_forward_count', 'reused_monitor_forward_count', 'extra_policy_monitor_forwards')
        if any(any(peer.get(name) != result.get(name) for name in shared) for peer in results):
            raise RuntimeError('Collective attempt dispositions or actual Adam counts differ')
    return result


def enforcement_for_role(role):
    if role not in ('production', DISCOVERY_ROLE, CONFIRMATION_ROLE):
        raise ValueError('Unknown policy safety execution role')
    return dict(negative_tail='observe_only' if role == DISCOVERY_ROLE else 'enforced',
                old_policy_kl='enforced', numerical_finite='enforced', dual_clip='enforced')


def dual_clip_terms(delta, advantages, mask, epsilon, c, *, check_finite=True):
    if not 0 < epsilon < 1 or not math.isfinite(c) or c <= 1 + epsilon:
        raise ValueError('Invalid Dual-Clip parameters')
    valid = mask.bool()
    d = torch.where(valid, delta, 0.).double()
    a = torch.where(valid, advantages, 0.).double()
    if check_finite and (not torch.isfinite(d).all() or not torch.isfinite(a).all()):
        raise ValueError('Nonfinite effective Dual-Clip input')
    bounded = torch.where(a > 0, d.clamp_max(math.log1p(epsilon)),
                          torch.where(a < 0, d.clamp(math.log1p(-epsilon), math.log(c)),
                                      torch.zeros_like(d)))
    return -a * torch.exp(bounded)


@torch.no_grad()
def monitor(delta, advantages, mask, weights, *, c, kl_log_bound, target_kl,
            negative_tail_limit, group=None, execution_role='production',
            finite_thresholds_diagnostic_only=False):
    """Global weighted k3 and exact negative-token quantiles, matching collectives.

    Invalid/over-bound samples generate finite diagnostics and a rejection flag;
    their substituted zero is NEVER reported as a true complete KL estimate.
    Accumulator fields are already global; identical rank reports describe one
    observation. The numerator is diagnostic when kl_estimate_complete is false.
    """
    enforcement = enforcement_for_role(execution_role)
    if type(finite_thresholds_diagnostic_only) is not bool:
        raise ValueError('Finite-threshold monitoring mode must be an explicit bool')
    if finite_thresholds_diagnostic_only:
        enforcement.update(negative_tail='observe_only',old_policy_kl='observe_only',log_ratio_bound='observe_only')
    tail_valid = negative_tail_limit is None if execution_role == DISCOVERY_ROLE else (
        type(negative_tail_limit) in (int, float) and math.isfinite(negative_tail_limit) and 0 <= negative_tail_limit <= 1)
    if not (math.isfinite(target_kl) and target_kl > 0 and tail_valid and
            math.isfinite(kl_log_bound) and kl_log_bound > 0 and math.isfinite(c) and c > 1):
        raise ValueError('Explicit finite calibrated thresholds required')
    v = mask.bool();d = torch.where(v, delta, 0.).double()
    a = torch.where(v, advantages, 0.).double();w = torch.where(v, weights, 0.).double()
    bad = v & (~torch.isfinite(d) | ~torch.isfinite(a) | ~torch.isfinite(w) | (w < 0))
    over = v & (d > kl_log_bound)
    safe_d = torch.where(bad | over, torch.zeros_like(d), d)
    safe_w = torch.where(bad, torch.zeros_like(w), w)
    negative = v & ~bad & (a < 0);tail = negative & (d > math.log(c))
    k3 = torch.expm1(safe_d) - safe_d
    values = torch.stack([safe_w.sum(), (k3 * safe_w).sum(),
                          safe_w[negative].sum(), safe_w[tail].sum(),
                          bad.sum().double(), over.sum().double(),v.sum().double(),
                          negative.sum().double(),tail.sum().double()])
    distributed = dist.is_initialized()
    if distributed: dist.all_reduce(values, group=group)
    mass, numerator, nmass, tmass, invalid, oversized, count, ncount, tcount = values.tolist()
    finite_totals = all(math.isfinite(x) for x in values.tolist())
    samples = d[negative & torch.isfinite(d)].flatten()
    if distributed:
        size = torch.tensor([samples.numel()], dtype=torch.long, device=d.device)
        sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size(group))]
        dist.all_gather(sizes, size, group=group)
        width = max(1, max(int(x.item()) for x in sizes))
        padded = torch.zeros(width, dtype=torch.float64, device=d.device)
        padded[:samples.numel()] = samples
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded, group=group)
        samples = torch.cat([x[:int(n.item())] for x,n in zip(gathered,sizes)])
    quantiles = {k: None for k in ('negative_log_ratio_p95','negative_log_ratio_p99','negative_log_ratio_max')}
    if samples.numel():
        quantiles.update(negative_log_ratio_p95=float(torch.quantile(samples,.95,interpolation='nearest')),
                         negative_log_ratio_p99=float(torch.quantile(samples,.99,interpolation='nearest')),
                         negative_log_ratio_max=float(samples.max()))
    complete = finite_totals and invalid == 0 and oversized == 0 and mass > 0 and count > 0
    kl = numerator / mass if complete else None
    fraction = tmass / nmass if finite_totals and nmass > 0 else 0.
    reasons = []
    if not finite_totals: reasons.append('nonfinite_global_totals')
    if invalid: reasons.append('nonfinite_or_negative_effective_input')
    if oversized: reasons.append('delta_above_kl_log_bound')
    if mass <= 0 or count <= 0: reasons.append('empty_effective_samples')
    if kl is not None and kl > target_kl: reasons.append('old_policy_kl')
    if negative_tail_limit is not None and fraction > negative_tail_limit: reasons.append('negative_advantage_tail')
    threshold_reasons = {'delta_above_kl_log_bound','old_policy_kl','negative_advantage_tail'}
    warnings = [reason for reason in reasons if finite_thresholds_diagnostic_only and reason in threshold_reasons]
    fatal = [reason for reason in reasons if reason not in warnings]
    return dict(execution_role=execution_role, enforcement=enforcement,
                finite_thresholds_diagnostic_only=finite_thresholds_diagnostic_only,
                diagnostic_warnings=warnings,fatal_reasons=fatal,
                allowed=not fatal,reasons=fatal,sampled_old_policy_kl=kl,
                kl_estimate_complete=complete,effective_weight_mass=mass if finite_totals else None,
                weighted_k3_numerator=numerator if finite_totals else None,
                negative_effective_weight_mass=nmass if finite_totals else None,
                tail_effective_weight_mass=tmass if finite_totals else None,
                effective_token_count=int(count),negative_token_count=int(ncount),
                negative_tail_count=int(tcount),negative_tail_count_fraction=tcount/ncount if ncount else 0.,
                negative_tail_weight_fraction=fraction,dual_clip_saturated_objective_weight_fraction=tmass/mass if finite_totals and mass>0 else None,
                invalid_count=int(invalid),overflow_count=int(oversized),quantile_method='exact global nearest-order statistic',**quantiles)


def bind_thresholds(config, *, group=None):
    """Reject missing/unbound calibration collectively before any update."""
    import json,hashlib,os
    from pathlib import Path
    errors=[];artifact={}
    try:
        from reproduction.common8.policy_calibration_plan import canonical_bytes,_strict_json
        base=Path(os.environ['SEARCH_G1_BASE']).resolve()
        path=Path(config.calibration_manifest).resolve()
        if path != base/'logs/policy-safety-calibration.json':raise ValueError('Unbound calibration path')
        artifact=_strict_json(path.read_bytes())
        run=_strict_json((base/'logs/run-manifest.json').read_bytes())
        if type(artifact['schema_version']) is not int or artifact['schema_version'] not in (2, 3, 4):
            raise ValueError('Explicit production admission or typed guarded fresh200 bootstrap required')
        # This reader verifies the frozen parent probability proof and the real
        # confirmation child proof with their distinct run IDs. It has no writes,
        # fresh/no-claim checks or live GPU/HTTP actions after training is claimed.
        if artifact['schema_version'] == 4:
            from reproduction.common8.formal_inline_safety import verify_plan
            tail,verified_sha=verify_plan(base,artifact,run)
        elif artifact['schema_version'] == 3:
            from reproduction.common8.fresh200_bootstrap import verify_bootstrap
            tail,verified_sha=verify_bootstrap(base,artifact,run)
        else:
            from reproduction.common8.production_safety_admission import verify_admission
            tail,verified_sha=verify_admission(base,artifact,run)
        if artifact['dual_clip']!=float(config.dual_clip) or artifact['target_old_policy_kl']!=float(config.target_old_policy_kl) or artifact['kl_log_bound']!=float(config.kl_log_bound):raise ValueError('Calibration thresholds differ')
        if type(tail) not in (int,float) or not math.isfinite(tail) or not 0<=tail<=1:
            raise ValueError('Invalid calibrated tail limit')
        if config.negative_tail_limit is not None and float(config.negative_tail_limit)!=tail:raise ValueError('Tail threshold differs')
        serialized=canonical_bytes(artifact)
        if hashlib.sha256(serialized).hexdigest()!=verified_sha:raise ValueError('Verified production admission hash differs')
    except Exception as failure:
        errors.append(str(failure));serialized=b'REJECTED'
    device='cuda:'+str(torch.cuda.current_device()) if dist.is_initialized() and dist.get_backend(group)=='nccl' else 'cpu'
    ok=torch.tensor(int(not errors),device=device,dtype=torch.int32)
    digest=torch.tensor(list(hashlib.sha256(serialized).digest()),device=device,dtype=torch.uint8)
    if dist.is_initialized():
        dist.all_reduce(ok,op=dist.ReduceOp.MIN,group=group)
        hashes=[torch.empty_like(digest) for _ in range(dist.get_world_size(group))]
        dist.all_gather(hashes,digest,group=group)
        if any(not torch.equal(x,digest) for x in hashes):ok.zero_()
    if not ok.item():raise RuntimeError('Policy safety calibration admission rejected on all ranks: '+repr(errors))
    return tail,hashlib.sha256(serialized).hexdigest()


def bind_actual_config(actor_config, group=None):
    """Every actor rank participates, including disabled ranks, before branching."""
    import json,hashlib,os
    from pathlib import Path
    from omegaconf import OmegaConf
    from reproduction.common8.topology import monitor_trajectories_per_rank
    ok=True
    monitor_count=4
    try:
        document=OmegaConf.to_container(actor_config,resolve=True)
        serialized=json.dumps(document,sort_keys=True,allow_nan=False).encode()
        s=document.get('policy_safety')
        from search_g1.actor_performance import lean_actor, validate_lean_binding
        lean = lean_actor(document)
        if s is not None and not isinstance(s.get('enabled'),bool):
            ok=False
        if os.environ.get('SEARCH_G1_PROFILE')=='common8':
            run=json.loads((Path(os.environ['SEARCH_G1_BASE'])/'logs/run-manifest.json').read_text())
            validate_lean_binding(document, run)
            monitor_count=monitor_trajectories_per_rank(run)
            if run.get('easy2560_selection') is True:
                from reproduction.common8.precision_contract import easy2560_native4
                from reproduction.common8.topology import actor_local_micro
                ok = ok and (easy2560_native4(run)
                    and document.get('ppo_micro_batch_size') == actor_local_micro(run)
                    and document.get('ppo_logical_micro_batch_size') == 4)
            diagnostic = run.get('finite_thresholds_diagnostic_only',False)
            configured_diagnostic = s.get('finite_thresholds_diagnostic_only',False) if s else False
            if type(diagnostic) is not bool or configured_diagnostic is not diagnostic:
                ok=False
            if diagnostic and (s is None or s.get('enabled') is not True or s.get('dual_clip')!=3. or document.get('clip_ratio')!=.2 or
                    (run.get('question_probe') or {}).get('probe_architecture')!='qwen2_decoder3_binary_mix_v1' or run.get('policy_calibration_role','production')!='production'):
                ok=False
            extra = run.get('extra_policy_monitor_forwards',True)
            configured_extra = s.get('extra_policy_monitor_forwards',True) if s else True
            if type(extra) is not bool or configured_extra is not extra:
                ok=False
            if not extra and not diagnostic and not lean:
                ok=False
            if run.get('policy_safety_required') is True and (s is None or s.get('enabled') is not True):
                ok=False
        if lean and os.environ.get('SEARCH_G1_PROFILE')!='common8':
            ok=False
        if s and s.get('extra_policy_monitor_forwards',True) is False and os.environ.get('SEARCH_G1_PROFILE')!='common8':
            ok=False
        if s and type(s.get('extra_policy_monitor_forwards',True)) is not bool:
            ok=False
        if s and s.get('finite_thresholds_diagnostic_only',False) and os.environ.get('SEARCH_G1_PROFILE')!='common8':
            ok=False
        if s and type(s.get('finite_thresholds_diagnostic_only',False)) is not bool:
            ok=False
        if s and s['enabled']:
            epsilon=float(document['clip_ratio']);c=float(s['dual_clip'])
            ok=ok and (isinstance(s['enabled'],bool) and 0<epsilon<1 and math.isfinite(c) and c>1+epsilon
                and math.isfinite(float(s['target_old_policy_kl'])) and s['target_old_policy_kl']>0
                and math.isfinite(float(s['kl_log_bound'])) and s['kl_log_bound']>0
                and s['monitor_trajectories_per_rank']==monitor_count
                and not document['use_dynamic_bsz'])
    except (ValueError,TypeError,KeyError,OSError,AssertionError):
        ok=False;serialized=b'INVALID'
    device='cuda:'+str(torch.cuda.current_device()) if dist.is_initialized() and dist.get_backend(group)=='nccl' else 'cpu'
    flag=torch.tensor(int(ok),device=device,dtype=torch.int32)
    digest=torch.tensor(list(hashlib.sha256(serialized).digest()),device=device,dtype=torch.uint8)
    if dist.is_initialized():
        dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=group)
        hashes=[torch.empty_like(digest) for _ in range(dist.get_world_size(group))]
        dist.all_gather(hashes,digest,group=group)
        if any(not torch.equal(digest,x) for x in hashes):flag.zero_()
    if not flag.item():raise RuntimeError('Actual actor/safety configuration differs or is invalid; all ranks rejected')


def _phase_agreement(document, errors, group, *, current_uuid=None):
    """Local I/O and validity errors reach the same collective on every rank."""
    import hashlib
    from reproduction.common8.policy_calibration_plan import canonical_bytes
    serialized = canonical_bytes(document) if not errors else b'REJECTED'
    device = 'cuda:'+str(torch.cuda.current_device()) if dist.is_initialized() and dist.get_backend(group)=='nccl' else 'cpu'
    ok = torch.tensor(int(not errors), dtype=torch.int32, device=device)
    digest = torch.tensor(list(hashlib.sha256(serialized).digest()), dtype=torch.uint8, device=device)
    if dist.is_initialized():
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        hashes = [torch.empty_like(digest) for _ in range(dist.get_world_size(group))]
        dist.all_gather(hashes, digest, group=group)
        if any(not torch.equal(x,digest) for x in hashes): ok.zero_()
        if current_uuid is not None:
            uuid_digest=torch.tensor(list(hashlib.sha256(current_uuid.encode()).digest()),dtype=torch.uint8,device=device)
            devices=[torch.empty_like(uuid_digest) for _ in hashes]
            dist.all_gather(devices,uuid_digest,group=group)
            if len({bytes(x.cpu().tolist()) for x in devices}) != 8: ok.zero_()
    if not ok.item(): raise RuntimeError('Calibration phase binding rejected on all ranks: '+repr(errors))


def reward_math_fingerprint(run, environ):
    """Hash bound reward implementation and only its ECO environment, never publish values."""
    import hashlib
    from pathlib import Path
    from reproduction.common8.policy_calibration_plan import document_sha256
    repo=Path(__file__).resolve().parents[1]
    names=('verl/trainer/main_ppo_eco.py','search_g1/signals.py',
           'verl/utils/reward_score/qa_em.py','search_r1/qa_protocol.py')
    hashes={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in names}
    if any(run['source_sha256'].get(name)!=sha for name,sha in hashes.items()):
        raise ValueError('Reward implementation changed after source binding')
    required=dict(ECO_TRUST_VARIANT='balanced',ECO_TRUST_SIGNAL='probe',ECO_PROBE_GATE='1',
                  ECO_BAL_GROUND='0.6',ECO_BAL_NOSEARCH_BONUS='0.5',ECO_W_COST='0.02',
                  ECO_WRONG_GROUND_SCALE='0',ECO_PROBE_BASE='0',ECO_PROBE_ALPHA_COST='0',
                  ECO_USE_CF='0',ECO_NO_SELF='1')
    if any(environ.get(name)!=value for name,value in required.items()):
        raise ValueError('Actual reward environment differs from the common8 recipe')
    eco={name:value for name,value in environ.items() if name.startswith('ECO_')}
    return document_sha256(dict(source_sha256=hashes,eco_environment=eco))


def _verify_phase_runtime(base,run,plan,actor_config):
    """Authenticate existing native/model/config/probe evidence before a child write."""
    import hashlib,json,os
    from pathlib import Path
    from omegaconf import OmegaConf
    from reproduction.common8.policy_calibration_plan import canonical_bytes
    from reproduction.common8.stage_retrieval import binding,memory_bytes
    from reproduction.common8.model_smoke import aggregate,coexistence
    from reproduction.common8.validation_identity import expected
    from reproduction.common8.native_allocation import cuda_uuids
    def check(condition,message):
        if not condition: raise ValueError(message)
    actual_run,job,allocation=binding(base)
    check(actual_run==run and memory_bytes(job['TRES'])>=512*2**30,'Actual native child allocation differs')
    check(dist.is_initialized() and dist.get_world_size()==8 and dist.get_backend()=='nccl',
          'Calibration requires actual eight-rank NCCL workers')
    actual_uuids=cuda_uuids(count=torch.cuda.device_count())
    current_uuid=actual_uuids[torch.cuda.current_device()]
    check(current_uuid in allocation['actual_cuda_runtime_uuids'] and
          len(allocation['actual_cuda_runtime_uuids'])==len(set(allocation['actual_cuda_runtime_uuids']))==8,
          'Actual worker CUDA UUID mapping differs')
    cfg=json.loads((base/'logs/resolved-training-config.json').read_text())
    document=cfg['resolved_config'];config_sha=hashlib.sha256(canonical_bytes(document)).hexdigest()
    check(cfg['run_id']==run['run_id'] and cfg['slurm_job_id']==job['JobId'] and
          config_sha==cfg['resolved_config_sha256']==plan['resolved_config_sha256'],
          'Actual resolved child recipe differs')
    raw=document['actor_rollout_ref'];a=raw['actor'];r=raw['ref'];roll=raw['rollout']
    check(document['data']['train_batch_size']==512 and document['algorithm']['adv_estimator']=='grpo' and
          document['trainer']['n_gpus_per_node']==8 and document['trainer']['nnodes']==1,
          'Actual phase global rollout mathematics differs')
    check(a['ppo_mini_batch_size']==128 and a['ppo_micro_batch_size']==16 and a['ppo_epochs']==1 and
          a['optim']['lr']==1e-6 and a['optim']['lr_warmup_steps_ratio']==0 and
          a['optim']['warmup_style']=='constant' and a['grad_clip']==1. and a['clip_ratio']==.2 and
          a['state_masking'] is True and a['use_kl_loss'] is True and
          a['kl_loss_coef']==.001 and a['kl_loss_type']=='low_var_kl' and
          a['use_dynamic_bsz'] is False and a['ulysses_sequence_parallel_size']==1,
          'Actual actor mathematics differs')
    check(roll['n']==1 and roll['n_agent']==5 and roll['temperature']==1 and roll['top_p']==1.0 and
          roll['log_prob_micro_batch_size']==32 and r['log_prob_micro_batch_size']==32 and
          roll['dtype']=='float32','Actual sampling/reference mathematics differs')
    for part in (a,r):
        check(part['compute_dtype']=='float32' and part['allow_tf32'] is False and
              part['fsdp_config']['model_dtype']=='fp32','Actual phase precision differs')
    check(all(a['fsdp_config']['mixed_precision'][k]=='fp32' for k in ('param_dtype','reduce_dtype','buffer_dtype')),
          'Actual FSDP precision differs')
    actual=OmegaConf.to_container(actor_config,resolve=True)
    for name in ('clip_ratio','grad_clip','state_masking','use_kl_loss','kl_loss_coef','kl_loss_type',
                 'compute_dtype','allow_tf32','ppo_epochs','use_dynamic_bsz','ulysses_sequence_parallel_size'):
        check(canonical_bytes(actual[name])==canonical_bytes(a[name]),'Live actor config differs: '+name)
    check(actual['ppo_mini_batch_size']==16 and actual['ppo_micro_batch_size']==2 and
          actual['optim']['lr']==1e-6 and actual['optim']['lr_warmup_steps_ratio']==0 and
          actual['optim']['warmup_style']=='constant','Live actor normalization/scheduler differs')
    check(actual['fsdp_config']==a['fsdp_config'],'Live actor FSDP config differs')
    _,_,ids=expected(base,'stage')
    for phase in ('initial','coexistence'):
        root=base/'logs'/('model-smoke-'+phase);content=(root/'complete.json').read_bytes();proof=json.loads(content)
        check(proof['accepted'] is True and proof['run_id']==run['run_id'] and
              proof['slurm_job_id']==job['JobId'] and proof['phase']==phase and
              type(proof['actual_grpo_updates']) is int and proof['actual_grpo_updates']==0,
              'Actual initial/coexistence model proof differs')
        ranks=[]
        for rank in range(8):
            data=(root/f'rank-{rank}.json').read_bytes()
            check(hashlib.sha256(data).hexdigest()==proof['rank_file_sha256'][str(rank)],'Actual rank proof changed')
            ranks.append(json.loads(data))
        check(ranks==proof['ranks'] and aggregate(ranks,run['run_id'],job['JobId'],phase,ids,
              allocation['actual_cuda_runtime_uuids'])==config_sha,'Actual rank model/config proof differs')
        if phase=='initial':
            check(hashlib.sha256(content).hexdigest()==plan['initial_probability_gate']['proof_sha256'],
                  'Frozen original probability proof differs')
    coexistence(base,run,job)  # Actual owned live service births and current HTTP encoder/version.
    live=json.loads((base/'artifacts/live/manifest.json').read_text());fingerprint=plan['initial_reward_fingerprint']
    check(type(live['round']) is int and live['round']==0 and live['run_id']==run['run_id'] and
          live['encoder_sha256']==fingerprint['encoder_sha256'] and
          live['head_sha256']==fingerprint['head_sha256'],'Actual initial reward identity differs')
    for name,key in (('commit_probe.pkl','commit'),('boundary_probe.pkl','boundary')):
        check(hashlib.sha256(Path(live[key]).read_bytes()).hexdigest()==fingerprint['head_sha256'][name],
              'Actual frozen probe head bytes changed')
    check(reward_math_fingerprint(run,os.environ)==fingerprint['reward_math_sha256'],
          'Actual reward mathematics differs')
    return current_uuid


def verify_fresh_phase_state(optimizer, logical_update, optimizer_writes):
    if type(logical_update) is not int or logical_update != 1 or type(optimizer_writes) is not int or optimizer_writes != 0:
        raise ValueError('A calibration child permits exactly its first fresh outer')
    if len(optimizer.state) != 0:
        raise ValueError('Calibration actor must have actually empty Adam state')
    if not optimizer.param_groups or any(type(group.get('lr')) not in (int,float) or group['lr'] != 1e-6
                                         for group in optimizer.param_groups):
        raise ValueError('Calibration actual optimizer initial LR must be 1e-6')


def calibration_state_snapshot(module,optimizer):
    """Actual local tensors and scalar Adam steps, without changing state."""
    fields={f'parameter_{i}':p.detach() for i,p in enumerate(module.parameters())}
    parameter_tensor_count=len(fields);parameter_numel=sum(x.numel() for x in fields.values())
    steps=[];state_tensor_count=0;state_numel=0
    for index,state in enumerate(optimizer.state.values()):
        step=state['step']
        if torch.is_tensor(step):
            if step.numel()!=1: raise ValueError('Adam step must be scalar')
            step=step.item()
        if type(step) not in (int,float) or not math.isfinite(step):
            raise ValueError('Adam step is not an actual finite scalar')
        steps.append(step)
        for name,value in state.items():
            if torch.is_tensor(value):
                fields[f'adam_{index}_{name}']=value
                state_tensor_count+=1;state_numel+=value.numel()
    return fields,dict(parameter_tensor_count=parameter_tensor_count,parameter_numel=parameter_numel,
        adam_state_entries=len(optimizer.state),adam_state_tensor_count=state_tensor_count,
        adam_state_numel=state_numel,adam_steps=steps)


def publish_calibration_optimizer_proof(module,optimizer,*,execution_role,plan_sha256,
                                        initial_writes,initial_state_entries,final_writes,group=None):
    """One independent durable rank proof after all real writes; no resume claim."""
    import json,os
    from pathlib import Path
    from search_g1.numerics import require_finite
    from reproduction.common8.native_allocation import cuda_uuids
    fields={};proof={};valid=True
    device='cuda:'+str(torch.cuda.current_device()) if dist.is_initialized() and dist.get_backend(group)=='nccl' else 'cpu'
    try:
        if execution_role not in (DISCOVERY_ROLE,CONFIRMATION_ROLE): raise ValueError('Not a calibration child')
        fields,snapshot=calibration_state_snapshot(module,optimizer)
        valid=(type(initial_writes) is int and initial_writes==0 and initial_state_entries==0 and
               type(final_writes) is int and final_writes==20 and snapshot['parameter_numel']>0 and
               snapshot['adam_state_entries']>0 and len(snapshot['adam_steps'])==snapshot['adam_state_entries'] and
               all(step==20 for step in snapshot['adam_steps']))
        base=Path(os.environ['SEARCH_G1_BASE']);run=json.loads((base/'logs/run-manifest.json').read_text())
        valid=valid and run['policy_calibration_role']==execution_role and run['policy_calibration_plan_sha256']==plan_sha256
        pid=os.getpid();ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        uuids=cuda_uuids(count=torch.cuda.device_count())
        proof=dict(run_id=run['run_id'],parent_production_run_id=run['parent_production_run_id'],
            execution_role=execution_role,plan_sha256=plan_sha256,rank=dist.get_rank(),
            slurm_job_id=os.environ['SLURM_JOB_ID'],pid=pid,start_ticks=ticks,
            cuda_uuid=uuids[torch.cuda.current_device()],initial_optimizer_writes=initial_writes,
            initial_adam_state_entries=initial_state_entries,final_optimizer_writes=final_writes,
            **snapshot,scope='actual local optimizer summary; not full-state resume or healthy admission')
    except Exception:
        valid=False
    fields['valid_actual_calibration_state']=torch.tensor(int(valid),device=device)
    require_finite(fields,stage='calibration_complete_actual_optimizer_state',process_group=group,
                   positive=('valid_actual_calibration_state',))
    proof.update(all_parameters_and_adam_finite=True)
    written=True
    try:
        target=base/'logs'/f'policy-calibration-optimizer-rank-{proof["rank"]}.json'
        with target.open('x') as f:
            json.dump(proof,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
        fd=os.open(str(target.parent),os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except Exception:
        written=False
    require_finite({'durable_calibration_state_proof':torch.tensor(int(written),device=device)},
                   stage='calibration_complete_actual_optimizer_proof_fsync',process_group=group,
                   positive=('durable_calibration_state_proof',))


def bind_policy_phase(actor_config, optimizer, *, logical_update, optimizer_writes, group=None):
    """Separate explicit child plans from unchanged production healthy admission.

    The first real update checks the live Adam state, not a JSON declaration.
    All ranks agree on their role before taking either admission path.
    """
    import json, os
    from pathlib import Path
    errors=[]; run={}
    try:
        base=Path(os.environ['SEARCH_G1_BASE']).resolve()
        run=json.loads((base/'logs/run-manifest.json').read_text())
        role=run.get('policy_calibration_role', 'production')
        enforcement_for_role(role)
        if role != 'production' and (run.get('profile') != 'common8' or run.get('policy_safety_required') is not True):
            raise ValueError('Calibration requires the explicit common8 safety manifest')
    except Exception as failure:
        errors.append(str(failure));role='invalid'
    _phase_agreement(dict(role=role),errors,group)
    if role == 'production':
        tail,sha=bind_thresholds(actor_config.policy_safety,group=group)
        return tail,sha,role
    errors=[];plan={};sha=None;current_uuid='INVALID'
    try:
        from reproduction.common8.policy_calibration_plan import load_plan
        plan,sha=load_plan(base,run,expected_slurm_job_id=os.environ['SLURM_JOB_ID'])
        verify_fresh_phase_state(optimizer,logical_update,optimizer_writes)
        current_uuid=_verify_phase_runtime(base,run,plan,actor_config)
        thresholds=plan['candidate_thresholds'];safety=actor_config.policy_safety
        for key in ('dual_clip','target_old_policy_kl','kl_log_bound'):
            if float(safety[key]) != thresholds[key]: raise ValueError('Actual phase threshold differs: '+key)
        if safety.negative_tail_limit is not None and safety.negative_tail_limit != thresholds['negative_tail_limit']:
            raise ValueError('Actual tail candidate differs from frozen phase')
    except Exception as failure:
        errors.append(str(failure))
    _phase_agreement(dict(role=role,plan=plan,sha256=sha),errors,group,current_uuid=current_uuid)
    return plan['candidate_thresholds']['negative_tail_limit'],sha,role
