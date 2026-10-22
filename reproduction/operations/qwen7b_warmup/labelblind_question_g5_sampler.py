"""One label-blind G5 per question, class-balanced global BCE by weights.

Question order and all five trajectory indices are independent of EM. Labels
only determine actual training targets and the two global loss weights. Every
question is visited exactly once, including the68-question tail.
"""
import copy
import math
import random
from .g10_grouping import _validate, _hash
from .balanced_hard_g10_sampler import _tuples


class LabelBlindQuestionG5Sampler:
    def __init__(self, groups, seed=2026102204, source_cursor=2577,
                 epoch_end_cursors=(3898, 5219), rng_state=None,
                 forbidden_qids=(), forbidden_hashes=()):
        blocked, hashes = set(forbidden_qids), set(forbidden_hashes)
        if any(g['qid'] in blocked or g.get('question_hash') in hashes for g in groups):
            raise ValueError('Independent evaluation identity cannot enter training')
        own = _validate(groups, True)
        self.groups = {g['qid']:g for g in own}
        self.all_qids = [g['qid'] for g in own]
        self.batch_questions = 128
        self.source_cursor, self.boundaries = source_cursor, tuple(epoch_end_cursors)
        self.sweep_steps = math.ceil(len(own)/128)
        if len(own) != 169028 or self.boundaries != tuple(source_cursor+self.sweep_steps*(i+1) for i in range(len(self.boundaries))):
            raise ValueError('Full169028 distinct-question epoch boundaries required')
        self.binding = _hash(dict(version='labelblind_own_G5_unique_global_class_weight_v1',
            groups=own, seed=seed, source_cursor=source_cursor,
            epoch_end_cursors=self.boundaries, rng_inherited=rng_state is not None,
            forbidden_qids=sorted(blocked), forbidden_hashes=sorted(hashes)))
        self.pool_qids_sha256 = _hash(self.all_qids)
        self.rng = random.Random(seed)
        if rng_state is not None:self.rng.setstate(_tuples(rng_state))
        self.seed = seed
        self.accepted_updates, self.epoch = source_cursor, 1
        self.pending, self._partition_cache, self.coverage_receipts = None, {}, []
        self._begin_epoch()

    @property
    def steps_per_epoch(self):
        return next((c for c in self.boundaries if c > self.accepted_updates), self.boundaries[-1]+1)

    def _indices(self, qid):
        # Neither EM nor dataset/class enters question-specific subset selection.
        return sorted(random.Random(_hash(dict(epoch_plan_seed=self.epoch_plan_seed, qid=qid))).sample(range(10), 5))

    def _begin_epoch(self):
        self.epoch_plan_seed = _hash(dict(inherited_entropy=self.rng.getrandbits(256),
                                         sampling_seed=self.seed, epoch=self.epoch))
        self.orders = {'all':list(self.all_qids)}
        self.rng.shuffle(self.orders['all'])
        self.positions, self.cycles = {'all':0}, {'all':self.epoch-1}
        mixed = sum(0 < sum(self.groups[q]['labels'][i] for i in self._indices(q)) < 5 for q in self.all_qids)
        self.class_counts = {'mixed':mixed, 'nonmixed':len(self.groups)-mixed}
        if not all(self.class_counts.values()):raise ValueError('Global class-balanced objective requires both actual classes')
        self.class_weights = {role:len(self.groups)/(2.*count) for role,count in self.class_counts.items()}

    def next_batch(self):
        if self.pending is not None or self.accepted_updates >= self.boundaries[-1]:
            raise ValueError('Only next unfinished accepted update may be sampled')
        before = dict(accepted_updates=self.accepted_updates, epoch=self.epoch,
                      position=self.positions['all'], epoch_plan_seed=self.epoch_plan_seed)
        start = self.positions['all']
        qids = self.orders['all'][start:start+128]
        self.positions['all'] += len(qids)
        views, roles = [], []
        for qid in qids:
            group = self.groups[qid]
            indices = self._indices(qid)
            labels = [group['labels'][i] for i in indices]
            K = sum(labels)
            role = 'mixed' if 0 < K < 5 else 'nonmixed'
            roles.append(role)
            views.append(dict(qid=qid, source_indices=indices,
                trajectory_ids=[group['trajectory_ids'][i] for i in indices], labels=labels,
                K=K, true_mixed=0 < K < 5, weight=1., weight_numerator=1,
                weight_denominator=1, loss_class_weight=self.class_weights[role],
                source_group_size=10))
        mixed = sum(v['true_mixed'] for v in views)
        summary = dict(mode='labelblind_own_G5_unique_global_BCE_classweight', questions=len(qids),
            views=len(qids), mixed_views=mixed, nonmixed_views=len(qids)-mixed,
            global_class_counts=self.class_counts, global_loss_class_weights=self.class_weights,
            class_balance_scope='epoch_weighted_objective_not_batch_counts',
            question_sampling_label_blind=True, G5_sampling_label_blind=True,
            hard_pool_used=False, views_per_question=1, per_question_total_weight_exactly_one=True,
            tail_batch=len(qids)<128, labels_recomputed_from_own_subset=True,
            one_visit_per_question_per_epoch=True, epoch=self.epoch)
        result = dict(source_qids=qids, roles=roles, grouping=dict(groups=views, summary=summary),
                      before=before, token=_hash(dict(before=before, qids=qids, views=views)))
        self.pending = copy.deepcopy(result)
        return result

    def observe(self, batch, losses):
        if self.pending != batch or len(losses) != len(batch['source_qids']) or any(not math.isfinite(float(x)) or x < 0 for x in losses):
            raise ValueError('Accepted label-blind G5/loss binding differs')
        self.accepted_updates += 1
        self.pending = None
        if self.positions['all'] == len(self.groups):
            if self.accepted_updates not in self.boundaries or sorted(self.orders['all']) != self.all_qids:
                raise ValueError('Every question must appear exactly once in a complete epoch')
            self.coverage_receipts.append(dict(epoch=self.epoch, end_cursor=self.accepted_updates,
                questions=len(self.groups), unique_questions=len(self.groups),
                mixed_views=self.class_counts['mixed'], nonmixed_views=self.class_counts['nonmixed'],
                class_weights=self.class_weights, qid_sha256=self.pool_qids_sha256,
                epoch_plan_seed=self.epoch_plan_seed, tail_batch_questions=68,
                one_real_G5_per_question=True, G5_sampling_label_blind=True, hard_pool_used=False))
            self.epoch += 1
            self._begin_epoch()

    def state_dict(self):
        if self.pending is not None:raise ValueError('Checkpoint requires an accepted update boundary')
        consumed = self.orders['all'][:self.positions['all']]
        return copy.deepcopy(dict(binding=self.binding, rng_state=self.rng.getstate(),
            orders=self.orders, positions=self.positions, cycles=self.cycles,
            accepted_updates=self.accepted_updates, epoch=self.epoch,
            epoch_plan_seed=self.epoch_plan_seed, class_counts=self.class_counts,
            class_weights=self.class_weights,
            seen_current_epoch=dict(count=len(consumed), unique_count=len(consumed), qid_sha256=_hash(sorted(consumed))),
            epoch_coverage_receipts=self.coverage_receipts))

    def load_state_dict(self, state):
        if state['binding'] != self.binding or set(state) != set(self.state_dict()):
            raise ValueError('Label-blind sampler binding/state differs')
        cursor = state['accepted_updates']
        completed, within = divmod(cursor-self.source_cursor, self.sweep_steps)
        if (not self.source_cursor <= cursor <= self.boundaries[-1]
            or state['epoch'] != 1+completed or len(state['epoch_coverage_receipts']) != completed
            or state['positions'] != {'all':within*128} or state['cycles'] != {'all':completed}
            or sorted(state['orders']['all']) != self.all_qids
            or sum(state['class_counts'].values()) != len(self.groups)):
            raise ValueError('Actual unique-question coverage/global class counts differ')
        self.rng.setstate(_tuples(state['rng_state']))
        for key in ('orders','positions','cycles','accepted_updates','epoch','epoch_plan_seed','class_counts','class_weights'):
            setattr(self, key, copy.deepcopy(state[key]))
        self.coverage_receipts = copy.deepcopy(state['epoch_coverage_receipts'])
        self.pending = None
        if self.state_dict()['seen_current_epoch'] != state['seen_current_epoch']:
            raise ValueError('Actual consumed unique-question prefix differs')
