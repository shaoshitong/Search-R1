"""Original inline PPO safeguards, with no admission scans or rollback.

Only adjacent after/before observations of the identical retained actor state
may be shared. All actual Adam writes survive a later ordinary policy guard.
"""
import hashlib
import json

REFERENCE_MICRO = 2
REFERENCE_MICROS_PER_VIRTUAL_MINI = 8
POLICY_SAFEGUARDS = dict(dual_clip=3., target_old_policy_kl=.2,
                         negative_tail_limit=.01, kl_log_bound=30.)


def tensor_identity(batch):
    import torch
    return {name: hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in batch.items()}


def frozen_monitor_ids(plan, rows):
    """Four length strata from each original virtual rank's 320 actual rows."""
    result = []
    for block in range(2):
        assigned = [i for mini in plan for i in mini[block * 16:(block + 1) * 16]]
        if len(assigned) != 320 or len(set(assigned)) != 320:
            raise ValueError('Monitor must use the exact physical4/reference8 rank assignment')
        ordered = sorted(assigned, key=lambda i: sum(rows[i]['response_info_mask']))
        # torch.linspace(0,319,4).long(), as in the original actor.
        result.extend(ordered[index] for index in (0, 106, 212, 319))
    return result


def reference_token_weights(mask):
    """Each frozen two-row micro has unit mass, including unequal lengths."""
    import torch
    if mask.ndim != 2 or mask.shape[0] % REFERENCE_MICRO:
        raise ValueError('Frozen monitor must contain complete reference micro2 pairs')
    weights = torch.zeros_like(mask, dtype=torch.float64)
    for offset in range(0, len(mask), REFERENCE_MICRO):
        pair = mask[offset:offset + REFERENCE_MICRO].double()
        weights[offset:offset + REFERENCE_MICRO] = pair / pair.sum().clamp_min(1.)
    return weights


def opportunity_result(*, opportunity, before, after, pre, post, numerical_guard=None):
    """Typed actual accounting; an after guard never undoes an Adam write."""
    if (type(opportunity) is not int or not 0 <= opportunity < 20 or
            type(before) is not int or type(after) is not int or after - before not in (0, 1)):
        raise ValueError('Invalid opportunity or actual returned Adam counter')
    wrote = after == before + 1
    if wrote and not pre['allowed']:
        raise ValueError('A rejected pre monitor cannot authorize an Adam write')
    if not wrote and pre['allowed'] and numerical_guard is None:
        raise ValueError('An unwritten healthy opportunity requires its actual numerical guard')
    return dict(opportunity=opportunity, processed=True,
        status=('written_with_post_guard' if not post['allowed'] else 'written') if wrote else 'skipped_guard',
        optimizer_write_succeeded=wrote, actual_adam_before=before, actual_adam_after=after,
        pre_allowed=pre['allowed'], post_allowed=post['allowed'],
        guard_events=[dict(stage=stage, reasons=report['reasons'])
                      for stage, report in (('before_minibatch', pre), ('after_minibatch', post))
                      if not report['allowed']] + ([numerical_guard] if numerical_guard else []),
        retained_partial_adam_writes=wrote, automatic_rollback_performed=False)


def attempt_result(opportunities, *, writes_before):
    if len(opportunities) != 20 or [row['opportunity'] for row in opportunities] != list(range(20)):
        raise ValueError('A returned attempt must process exactly20 actual opportunities')
    cursor = writes_before
    for row in opportunities:
        if (not row['processed'] or row['actual_adam_before'] != cursor or
                type(row['actual_adam_after']) is not int or row['actual_adam_after'] - cursor not in (0,1) or
                row['optimizer_write_succeeded'] != (row['actual_adam_after'] == cursor + 1)):
            raise ValueError('Actual opportunity counters are not contiguous')
        cursor = row['actual_adam_after']
    writes = cursor - writes_before
    return dict(status='complete' if writes else 'recoverable_guard', complete_outer=bool(writes),
        full_budget_complete=writes == 20, opportunities_processed=20,
        optimizer_writes_before=writes_before, optimizer_writes_after=cursor,
        optimizer_writes_this_attempt=writes, skipped_update_opportunities=20 - writes,
        written_with_post_guard=sum(row['status'] == 'written_with_post_guard' for row in opportunities),
        opportunity_statuses=[row['status'] for row in opportunities],
        scheduler_advances=int(bool(writes)), retained_partial_adam_writes=bool(writes),
        automatic_rollback_performed=False)


class FrozenPolicyMonitor:
    """Inline frozen-sample monitor, bound to actor state, input and thresholds."""
    def __init__(self, actor, batch, *, device, local_micro, identity):
        self.actor, self.batch, self.device, self.local_micro = actor, batch, device, local_micro
        self.weights = reference_token_weights(batch['loss_mask'])
        self.tensor_sha256 = tensor_identity(batch)
        self.binding = dict(identity, frozen_tensor_sha256=self.tensor_sha256,
                            thresholds=dict(POLICY_SAFEGUARDS), temperature=1.,
                            compute_dtype='bfloat16', objective_weights='equal reference micro2 masked means')
        self.binding_sha256 = hashlib.sha256(json.dumps(self.binding, sort_keys=True).encode()).hexdigest()
        self.cached = None
        self.actual_forward_calls = self.fresh_observations = self.reused_observations = 0

    def observe(self, *, stage, opportunity, actual_adam_steps):
        import torch
        import torch.distributed as dist
        from search_g1.policy_safety import monitor
        if stage not in ('before_minibatch', 'after_minibatch'):
            raise ValueError('Unknown original policy monitor stage')
        binding_sha256 = hashlib.sha256(json.dumps(self.binding, sort_keys=True).encode()).hexdigest()
        valid = (tensor_identity(self.batch) == self.tensor_sha256 and
                 binding_sha256 == self.binding_sha256 and self.binding['thresholds'] == POLICY_SAFEGUARDS)
        agreed = torch.tensor(int(valid), dtype=torch.int32, device=self.device)
        if dist.is_initialized(): dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
        if not agreed.item(): raise ValueError('Frozen monitor inputs changed or bound monitor thresholds/identity changed')
        key = (self.binding_sha256, actual_adam_steps)
        can_reuse = (stage == 'before_minibatch' and self.cached is not None and
                     self.cached['key'] == key and self.cached['opportunity'] == opportunity - 1)
        # All ranks must choose the same collective path, even if a cache is lost.
        agreed.fill_(int(can_reuse))
        if dist.is_initialized(): dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
        reused = bool(agreed.item())
        if reused:
            report = dict(self.cached['report']); self.reused_observations += 1
        else:
            was_training = self.actor.actor_module.training
            self.actor.actor_module.eval()
            values = []
            try:
                with torch.no_grad():
                    for offset in range(0, len(self.batch['responses']), self.local_micro):
                        micro = {name: value[offset:offset + self.local_micro].to(self.device)
                                 for name, value in self.batch.items()}
                        _, new = self.actor._forward_micro_batch(micro, temperature=1., compute_entropy=False)
                        authored = micro['loss_mask'].bool()
                        values.append(torch.where(authored, new, 0.).double() -
                                      torch.where(authored, micro['old_log_probs'], 0.).double())
                        self.actual_forward_calls += 1
                report = monitor(torch.cat(values), self.batch['advantages'].to(self.device),
                    self.batch['loss_mask'].to(self.device), self.weights.to(self.device),
                    c=self.binding['thresholds']['dual_clip'], kl_log_bound=self.binding['thresholds']['kl_log_bound'],
                    target_kl=self.binding['thresholds']['target_old_policy_kl'],
                    negative_tail_limit=self.binding['thresholds']['negative_tail_limit'])
                self.fresh_observations += 1
            finally:
                self.actor.actor_module.train(was_training)
        if stage == 'after_minibatch':
            self.cached = dict(key=key, opportunity=opportunity, report=dict(report))
        return dict(report, stage=stage, opportunity=opportunity, actual_adam_steps=actual_adam_steps,
            binding=self.binding, binding_sha256=self.binding_sha256,
            reused_adjacent_post_observation=reused,
            source_stage='after_minibatch' if reused else stage,
            source_opportunity=opportunity - 1 if reused else opportunity,
            actual_monitor_forward_calls=self.actual_forward_calls,
            fresh_monitor_observations=self.fresh_observations,
            reused_monitor_observations=self.reused_observations,
            monitoring_scope='four length-stratified trajectories per virtual rank;32 global;not full rollout')
