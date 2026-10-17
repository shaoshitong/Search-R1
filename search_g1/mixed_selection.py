"""Ordinal selection of complete strict-EM groups; no reward/probe side effects."""
import hashlib
import json
import unicodedata
from dataclasses import dataclass


class InsufficientEligibleGroups(RuntimeError):
    pass


def question_identity(question):
    text = ' '.join(unicodedata.normalize('NFKC', question).split())
    if not text:
        raise ValueError('Empty original question')
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    ordinal: int
    question_id: str
    content_sha256: str
    trajectory_ids: tuple
    correctness: tuple

    @property
    def category(self):
        count = sum(self.correctness)
        return 'all_correct' if count == len(self.correctness) else 'all_wrong' if count == 0 else 'mixed'


class MixedFirstSelector:
    def __init__(self, target=512, limit=2048, responses=5, fixed_pool=False):
        if not 0 < target <= limit or responses < 2:
            raise ValueError('Invalid group budget')
        self.target, self.limit, self.responses = target, limit, responses
        self.fixed_pool = bool(fixed_pool)
        self.reserved = []
        self.ids, self.contents, self.results, self.trajectories = set(), set(), {}, set()

    def reserve(self, question_id, question):
        """Called before generation; duplicates consume no candidate quota."""
        key, content = str(question_id), question_identity(question)
        if key in self.ids or content in self.contents:
            return None
        if len(self.reserved) >= self.limit or (not self.fixed_pool and self.mixed_count >= self.target):
            return None
        ordinal = len(self.reserved)
        self.ids.add(key); self.contents.add(content)
        self.reserved.append((key, content))
        return ordinal

    def complete(self, ordinal, trajectory_ids, correctness):
        if ordinal in self.results or not 0 <= ordinal < len(self.reserved):
            raise ValueError('Unreserved or repeated completion')
        ids, values = tuple(trajectory_ids), tuple(correctness)
        if len(ids) != self.responses or len(set(ids)) != self.responses or any(x in self.trajectories for x in ids):
            raise ValueError('Incomplete or duplicate real trajectories')
        if len(values) != self.responses or any(type(x) is not int or x not in (0, 1) for x in values):
            raise ValueError('Incomplete strict correctness vector')
        key, content = self.reserved[ordinal]
        self.results[ordinal] = Candidate(ordinal, key, content, ids, values)
        self.trajectories.update(ids)

    @property
    def mixed_count(self):
        # Only a complete ordinal prefix can trigger early stopping. Out-of-order
        # completions never choose a faster worker over a preceding candidate.
        prefix = 0
        while prefix in self.results:
            prefix += 1
        return sum(self.results[i].category == 'mixed' for i in range(prefix))

    def select(self):
        if len(self.results) != len(self.reserved):
            raise ValueError('Incomplete/error groups remain in flight')
        if self.fixed_pool and len(self.reserved) != self.limit:
            raise ValueError('Fixed selection requires every distinct candidate to complete')
        ordered = [self.results[i] for i in range(len(self.reserved))]
        mixed = [x for x in ordered if x.category == 'mixed']
        if len(mixed) >= self.target:
            members = mixed[:self.target]
        else:
            if len(ordered) != self.limit:
                raise ValueError('Fallback requires the full distinct candidate limit')
            correct = [x for x in ordered if x.category == 'all_correct']
            if len(mixed) + len(correct) < self.target:
                raise InsufficientEligibleGroups('INSUFFICIENT_ELIGIBLE_GROUPS')
            members = mixed + correct[:self.target - len(mixed)]
        return sorted(members, key=lambda x: x.ordinal)

    def audit(self):
        members = self.select()
        selected = {x.ordinal for x in members}
        return {'attempted_candidates': len(self.reserved), 'responses_per_group': self.responses,
                'selected_groups': len(members), 'selected_mixed': sum(x.category == 'mixed' for x in members),
                'selected_all_correct_fallback': sum(x.category == 'all_correct' for x in members),
                'groups': [dict(ordinal=x.ordinal, question_id=x.question_id, content_sha256=x.content_sha256,
                               trajectory_ids=x.trajectory_ids, strict_correctness=x.correctness,
                               category=x.category, selected=x.ordinal in selected,
                               source=('selected_' + ('mixed' if x.category == 'mixed' else 'all_correct_fallback'))
                               if x.ordinal in selected else 'excluded')
                           for x in sorted(self.results.values(), key=lambda x: x.ordinal)]}
