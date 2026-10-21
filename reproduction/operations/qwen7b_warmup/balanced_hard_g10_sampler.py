"""Training-only, stateful 1:1 real-G5 sampling from authentic G10 questions.

The producer's rollout/EM/feature receipts must be verified by the caller.
Conditional-on-class subset sampling deliberately changes the train prior;
evaluation must retain its own label-blind five IDs and natural class prior.
Hardness is observed training Mixed BCE, never a validation prediction.
"""
import copy
import math
import random
from .g10_grouping import _validate, _partitions, _hash


ROLES = ('mixed', 'nonmixed')


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, (list, tuple)) else value


class BalancedHardG10Sampler:
    """Q/2 questions/class; four real class-conditional views/question.

    Questions are distinct within a batch. Each occurrence has total weight1.
    Uniform shuffled cycles maintain coverage; after epoch0, half each class's
    slots request its previous training-observed top-quartile difficult pool.
    The other half remains uniform. Repetition across updates is intentional.
    Four subset draws are independent with replacement: K10=5 has only TWO
    nonMixed subsets, so four distinct negative views would be impossible.
    Every individual subset still contains five distinct authentic trajectories.
    """
    def __init__(self, groups, seed=1729, batch_questions=32, steps_per_epoch=None,
                 role='train', forbidden_qids=(), forbidden_hashes=()):
        if role != 'train':
            raise ValueError('This label-aware sampler is training-only')
        if type(batch_questions) is not int or batch_questions < 4 or batch_questions % 4:
            raise ValueError('Question batch must be a positive multiple of4')
        blocked, hashes = set(forbidden_qids), set(forbidden_hashes)
        if any(g['qid'] in blocked or g.get('question_hash') in hashes for g in groups):
            raise ValueError('Evaluation question identity cannot enter training sampling')
        own = _validate(groups, True)
        self.groups = {g['qid']: g for g in own}
        self.pools = {'mixed': [g['qid'] for g in own if 0 < sum(g['labels']) < 10],
                      'nonmixed': [g['qid'] for g in own]}
        if len(own) < batch_questions or len(self.pools['mixed']) < batch_questions // 2:
            raise ValueError('Insufficient genuine Mixed-capable questions for distinct balanced batch')
        if type(seed) is not int:
            raise ValueError('Explicit integer sampler seed required')
        self.batch_questions = batch_questions
        self.steps_per_epoch = steps_per_epoch or math.ceil(len(own) / batch_questions)
        if type(self.steps_per_epoch) is not int or self.steps_per_epoch < 1:
            raise ValueError('Positive integer steps_per_epoch required')
        self.binding = _hash(dict(version=1, groups=own, seed=seed,
            batch_questions=batch_questions, steps_per_epoch=self.steps_per_epoch,
            forbidden_qids=sorted(blocked), forbidden_hashes=sorted(hashes),
            hard_fraction='1/2', hard_top_fraction='1/4', views_per_question=4))
        self.rng = random.Random(seed)
        self.orders = copy.deepcopy(self.pools)
        for values in self.orders.values():
            self.rng.shuffle(values)
        self.positions = {r: 0 for r in ROLES}
        self.cycles = {r: 0 for r in ROLES}
        self.difficulty = {r: {} for r in ROLES}
        self.hard_pools = {r: [] for r in ROLES}
        self.accepted_updates = 0
        self.epoch = 0
        self.pending = None
        self._partition_cache = {}

    def _uniform(self, role, used):
        values = self.orders[role]
        for _ in range(2 * len(values) + self.batch_questions):
            if self.positions[role] == len(values):
                self.rng.shuffle(values)
                self.positions[role] = 0
                self.cycles[role] += 1
            qid = values[self.positions[role]]
            self.positions[role] += 1
            if qid not in used:
                return qid
        raise RuntimeError('Distinct balanced source questions cannot be sampled')

    def _question(self, role, used, request_hard):
        pool = self.hard_pools[role]
        if request_hard and pool:
            for _ in range(32):
                qid = self.rng.choice(pool)
                if qid not in used:
                    return qid, True
        return self._uniform(role, used), False

    def _subsets(self, group, role):
        pattern = tuple(group['labels'])
        if pattern not in self._partition_cache:
            self._partition_cache[pattern] = _partitions(group)
        pool = self._partition_cache[pattern][0 if role == 'mixed' else 1]
        if not pool:
            raise ValueError('A homogeneous question cannot fabricate a Mixed view')
        return [self.rng.choice(pool) for _ in range(4)]

    def next_batch(self):
        if self.pending is not None:
            raise ValueError('Previous sampled batch needs accepted-update observe before sampling again')
        before = dict(accepted_updates=self.accepted_updates, epoch=self.epoch,
                      rng_sha256=_hash(self.rng.getstate()), positions=dict(self.positions))
        qids, roles, views, per_question = [], [], [], []
        hard_count = {r: 0 for r in ROLES}
        used = set()
        for role in ROLES:
            for slot in range(self.batch_questions // 2):
                request_hard = self.epoch > 0 and slot < self.batch_questions // 4
                qid, hard = self._question(role, used, request_hard)
                used.add(qid); qids.append(qid); roles.append(role)
                hard_count[role] += int(hard)
                g = self.groups[qid]
                chosen = self._subsets(g, role)
                for indices in chosen:
                    labels = [g['labels'][i] for i in indices]
                    k = sum(labels)
                    assert (0 < k < 5) == (role == 'mixed')
                    views.append(dict(qid=qid, source_indices=list(indices),
                        trajectory_ids=[g['trajectory_ids'][i] for i in indices],
                        labels=labels, K=k, true_mixed=0 < k < 5,
                        weight_numerator=1, weight_denominator=4, weight=.25,
                        source_group_size=10))
                per_question.append(dict(qid=qid, role=role, views=4,
                    unique_subsets=len(set(chosen)), hard_pool_selected=hard,
                    total_weight_numerator=1, total_weight_denominator=1))
        self.rng.shuffle(views)
        n = self.batch_questions
        summary = dict(mode='training_balanced_hard_real_G10', questions=n,
            covered_questions=n, views=4*n, mixed_views=2*n, nonmixed_views=2*n,
            count_balanced=True, weighted_class_mass_balanced=True,
            weighted_mixed_fraction=.5, per_question_total_weight_exactly_one=True,
            four_subset_draws_with_replacement=True,
            repeated_subset_draws=sum(4-p['unique_subsets'] for p in per_question),
            synthetic_labels=False, homogeneous_questions_can_create_Mixed=False,
            labels_recomputed_from_own_subset=True, training_only=True,
            hard_pool_actual_counts=hard_count, epoch=self.epoch,
            source_question_pool_size=len(self.groups),
            claim_every_question_once_per_epoch=False,
            selected_views_sha256=_hash(views))
        token = _hash(dict(cursor=self.accepted_updates, qids=qids, views=views))
        out = dict(source_qids=qids, roles=roles,
                   grouping=dict(groups=views, per_question=per_question, summary=summary),
                   before=before, token=token)
        self.pending = copy.deepcopy(out)
        return out

    def observe(self, batch, per_view_mixed_BCE):
        """Call only after one actual complete accepted optimizer update."""
        if self.pending is None or batch != self.pending:
            raise ValueError('Only this exact pending training batch can update hardness')
        losses = list(per_view_mixed_BCE)
        views = batch['grouping']['groups']
        if len(losses) != len(views) or any(not math.isfinite(float(x)) or float(x) < 0 for x in losses):
            raise ValueError('Finite nonnegative training per-view Mixed BCE required')
        by = {q: [] for q in batch['source_qids']}
        for view, loss in zip(views, losses):
            by[view['qid']].append(float(loss))
        for qid, role in zip(batch['source_qids'], batch['roles']):
            if len(by[qid]) != 4:
                raise ValueError('Exactly four observations for each own source question')
            self.difficulty[role][qid] = sum(by[qid]) / 4
        self.accepted_updates += 1
        epoch = self.accepted_updates // self.steps_per_epoch
        if epoch != self.epoch:
            self.epoch = epoch
            for role in ROLES:
                ranked = sorted(self.difficulty[role], key=lambda q: (-self.difficulty[role][q], q))
                self.hard_pools[role] = ranked[:max(1, math.ceil(len(ranked)/4))]
        self.pending = None

    def state_dict(self):
        if self.pending is not None:
            raise ValueError('Sampler state is durable only after accepted update observation')
        return copy.deepcopy(dict(binding=self.binding, rng_state=self.rng.getstate(),
            orders=self.orders, positions=self.positions, cycles=self.cycles,
            difficulty=self.difficulty, hard_pools=self.hard_pools,
            accepted_updates=self.accepted_updates, epoch=self.epoch))

    def load_state_dict(self, state):
        expected = set(self.state_dict()) if self.pending is None else {
            'binding','rng_state','orders','positions','cycles','difficulty',
            'hard_pools','accepted_updates','epoch'}
        if set(state) != expected or state['binding'] != self.binding:
            raise ValueError('Sampler source/config/exclusion binding differs')
        if (type(state['accepted_updates']) is not int or state['accepted_updates'] < 0 or
                state['epoch'] != state['accepted_updates'] // self.steps_per_epoch):
            raise ValueError('Sampler epoch/cursor mismatch')
        for role in ROLES:
            if (len(state['orders'][role]) != len(self.pools[role]) or
                    set(state['orders'][role]) != set(self.pools[role]) or
                    not 0 <= state['positions'][role] <= len(self.pools[role]) or
                    not set(state['difficulty'][role]) <= set(self.pools[role]) or
                    not set(state['hard_pools'][role]) <= set(state['difficulty'][role]) or
                    any(not math.isfinite(float(x)) or float(x) < 0
                        for x in state['difficulty'][role].values())):
                raise ValueError('Sampler coverage or training-only difficulty state invalid')
        self.rng.setstate(_tuples(state['rng_state']))
        for key in ('orders','positions','cycles','difficulty','hard_pools','accepted_updates','epoch'):
            setattr(self, key, copy.deepcopy(state[key]))
        self.pending = None
