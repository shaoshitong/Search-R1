"""Resumeable funding of original G5 continuations, not extra reward.

The JSON ledger contains identities, scores, binary outcomes and remote payload
references only. Tensor/token payloads stay in the existing rollout artifacts.
Call ``commit_batch`` as completed G5s become durable. On restart ``next_batch``
returns only unfinished members of the existing reservation. It never resamples
questions, scores or prefixes.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from .selection import Candidate, _unique, group_counts, score_order


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%s" % os.getpid())
    with temporary.open("w") as stream:
        json.dump(data, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class BackstopLedger:
    def __init__(self, directory, candidates, initial_selected, *, outer,
                 actor_version, probe_version, batch_size=128, target=154):
        self.path = Path(directory) / "backstop-ledger.json"
        rows = _unique(candidates)
        self.pool = {row.question_id: row for row in rows}
        selected = _unique(initial_selected)
        if len(selected) != 512 or not {row.question_id for row in selected} <= self.pool.keys():
            raise ValueError("Backstop starts with the actual 512 selected original questions")
        for row in selected:
            original = self.pool[row.question_id]
            if (row.ordinal, row.score, row.strict_em) != (original.ordinal, original.score, original.strict_em):
                raise ValueError("Initial selection must retain the original score, ordinal and G5")
        if target != 154 or batch_size != 128:
            raise ValueError("Formal7B funding uses target154 and batches128")
        frozen = [[row.question_id, row.ordinal, row.score] for row in rows]
        binding = dict(outer=outer, actor_version=actor_version, probe_version=probe_version,
                       batch_size=batch_size, target=target,
                       pool_sha256=hashlib.sha256(json.dumps(frozen, separators=(",", ":")).encode()).hexdigest(),
                       initial_selected=[row.question_id for row in selected])
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["binding"] != binding:
                raise ValueError("A funding reservation cannot change actor, probe, pool or scores")
            # The ledger's completed G5s are authoritative on restore, even if
            # the caller initially loaded only its original prefix manifest.
            for row in rows:
                old = self.state["outcomes"].get(row.question_id)
                if old is not None and row.strict_em is not None and old != list(row.strict_em):
                    raise ValueError("Completed original G5 outcomes changed")
        else:
            outcomes = {row.question_id: list(row.strict_em) for row in rows if row.strict_em is not None}
            if any(row.question_id not in outcomes for row in selected):
                raise ValueError("All selected G5s must finish before funding")
            self.state = dict(schema_version=1, binding=binding, outcomes=outcomes,
                              artifacts={}, selected=[row.question_id for row in selected],
                              reserved=None, batches=[], replacements=[], cursor=0,
                              triggered=group_counts(selected)["mixed"] <= 153)
            if self.state["triggered"]:
                self._use_completed_mixed()
            self._save()

    def _row(self, question_id):
        row = self.pool[question_id]
        em = self.state["outcomes"].get(question_id)
        payload = self.state["artifacts"].get(question_id, row.payload)
        return replace(row, strict_em=tuple(em) if em is not None else None, payload=payload)

    def _save(self):
        _atomic_json(self.path, self.state)

    def selected(self):
        return [self._row(question_id) for question_id in self.state["selected"]]

    def _use_completed_mixed(self):
        selected_ids = self.state["selected"]
        selected_set = set(selected_ids)
        available = sorted((self._row(qid) for qid in self.state["outcomes"]
                            if qid not in selected_set and self._row(qid).outcome == "mixed"),
                           key=score_order)
        for incoming in available:
            current = self.selected()
            if group_counts(current)["mixed"] >= 154:
                break
            replaceable = [row for row in current if row.outcome == "all_wrong"]
            if not replaceable:
                replaceable = [row for row in current if row.outcome == "all_correct"]
            outgoing = min(replaceable, key=lambda row: (row.score, row.ordinal))
            position = selected_ids.index(outgoing.question_id)
            selected_ids[position] = incoming.question_id
            self.state["replacements"].append(dict(out=outgoing.question_id,
                                                    incoming=incoming.question_id,
                                                    position=position))

    def next_batch(self):
        """Return an existing unfinished reservation or reserve the next128.

        A completed partial reservation stays completed on resume. Its recorded
        payload is returned through ``completed`` and must be reloaded rather
        than generated again.
        """
        reserved = self.state["reserved"]
        if reserved is not None:
            return [self.pool[qid] for qid in reserved if qid not in self.state["outcomes"]]
        if not self.state["triggered"] or group_counts(self.selected())["mixed"] >= 154:
            return []
        remaining = sorted((row for row in self.pool.values()
                            if row.question_id not in self.state["outcomes"]), key=score_order)
        batch = remaining[:128]
        if batch:
            self.state["reserved"] = [row.question_id for row in batch]
            self._save()
        return batch

    def commit_batch(self, completed, artifact_refs=None):
        """Commit any finished subset, then replace only after the batch ends.

        ``artifact_refs`` maps question_id to its durable original G5 payload
        reference. The rollout writer finishes that payload before this call.
        Idempotent commits are accepted only with the same strict EM.
        """
        rows = _unique(completed)
        reserved = self.state["reserved"]
        if reserved is None:
            raise ValueError("No original-candidate batch is reserved")
        refs = artifact_refs or {}
        for row in rows:
            if row.question_id not in reserved:
                raise ValueError("Cannot commit an unreserved question")
            original = self.pool[row.question_id]
            if (row.ordinal, row.score) != (original.ordinal, original.score):
                raise ValueError("Funding uses the frozen original score and ordinal")
            row.outcome  # require a completed binary G5
            old = self.state["outcomes"].get(row.question_id)
            if old is not None and old != list(row.strict_em):
                raise ValueError("Cannot replace a completed G5 by a new sample")
            self.state["outcomes"][row.question_id] = list(row.strict_em)
            if row.question_id in refs:
                self.state["artifacts"][row.question_id] = refs[row.question_id]
        batch_done = all(qid in self.state["outcomes"] for qid in reserved)
        if batch_done:
            before = list(self.state["selected"])
            self._use_completed_mixed()
            self.state["batches"].append(dict(question_ids=reserved,
                                               selected_before=before,
                                               selected_after=list(self.state["selected"])))
            self.state["cursor"] += len(reserved)
            self.state["reserved"] = None
        self._save()
        return batch_done

    def completed(self):
        return [self._row(qid) for qid in self.state["outcomes"]]

    def summary(self):
        counts = group_counts(self.selected())
        remaining = len(self.pool) - len(self.state["outcomes"])
        return dict(triggered=self.state["triggered"], target=154, **counts,
                    completed_questions=len(self.state["outcomes"]),
                    additional_questions=self.state["cursor"],
                    pending_batch_questions=len(self.state["reserved"] or []),
                    remaining_questions=remaining,
                    shortfall=max(0, 154 - counts["mixed"]) if self.state["triggered"] and remaining == 0 else 0,
                    exhausted=remaining == 0)
