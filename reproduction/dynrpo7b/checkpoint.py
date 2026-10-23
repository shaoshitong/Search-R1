"""Small phase journal around the existing rolling/full-state actor saver.

This module never saves model tensors. ``actor_ref`` is the existing runtime's
committed four-rank model/Adam/scheduler/RNG receipt; ``driver_ref`` references
data order/cursor and RNG; ``probe_ref`` references raw/EMA/Adam/RNG. Their
actual writer performs the file saves and hashes once. This journal preserves
those receipts and the asymmetry between a saved actor and a pending probe, so
recovering a probe fit never replays a completed actor outer.
"""
import json
from pathlib import Path

from .mixed_backstop import _atomic_json


def probe_update_due(outer):
    return 0 < outer < 200 and outer % 10 == 0


def fitting_prefix(outer):
    return 80 if outer < 100 else 160


class RuntimeCheckpoint:
    def __init__(self, directory, run_id):
        self.path = Path(directory) / "formal-state.json"
        self.run_id = run_id
        self.state = json.loads(self.path.read_text()) if self.path.exists() else None
        if self.state is not None and self.state["run_id"] != run_id:
            raise ValueError("A phase journal belongs to one independent run")

    def _save(self):
        _atomic_json(self.path, self.state)

    def _phase(self):
        state = self.state
        if state["probe_pending"]:
            return "actor_complete_probe_pending"
        if state["grounding_pending"]:
            return "actor_complete_grounding_pending"
        return "probe_published" if probe_update_due(state["completed_actor_outer"]) else "outer_complete"

    def actor_complete(self, outer, actor_ref, driver_ref, probe_ref, *,
                       prefix_tokens, rollout_ref, grounding_version):
        """Call only after the existing actor full-state writer commits.

        References may be paths or JSON receipts; include all four rank files,
        their actual Adam-write counts, scheduler and RNG in ``actor_ref``.
        Do not pass a synthetic checkpoint path or an uncommitted save intent.
        """
        if not 1 <= outer <= 200:
            raise ValueError("Formal student budget is200 complete outers")
        expected_prefix = 80 if outer <= 100 else 160
        if prefix_tokens != expected_prefix:
            raise ValueError("Completed outer selection prefix does not match its phase")
        if self.state is not None and self.state["completed_actor_outer"] == outer:
            if (self.state["actor_ref"], self.state["driver_ref"], self.state["rollout_ref"]) != (
                    actor_ref, driver_ref, rollout_ref):
                raise ValueError("Cannot overwrite a completed actor with a different snapshot")
            return self.state
        if self.state is not None:
            if self.state["probe_pending"] or self.state["grounding_pending"]:
                raise ValueError("Publish the pending probe/reward version before the next actor")
            if self.state["completed_actor_outer"] + 1 != outer:
                raise ValueError("Completed actor outers must advance in sequence")
        self.state = dict(schema_version=1, run_id=self.run_id,
                          completed_actor_outer=outer, rollout_actor_outer=outer - 1,
                          actor_ref=actor_ref, driver_ref=driver_ref, probe_ref=probe_ref,
                          rollout_ref=rollout_ref, selection_prefix_tokens=prefix_tokens,
                          fitting_prefix_tokens=fitting_prefix(outer),
                          next_outer=outer + 1, probe_effective_from_outer=None,
                          probe_pending=probe_update_due(outer), probe_completed_epoch=0,
                          grounding_pending=outer in (50, 100, 150),
                          grounding_version=grounding_version)
        self.state["phase"] = self._phase()
        self._save()
        return self.state

    def probe_epoch(self, epoch, probe_ref):
        """An already saved epoch is the exact resume point, including Adam/RNG."""
        if not self.state or not self.state["probe_pending"]:
            raise ValueError("No saved actor is waiting for a probe fit")
        previous = self.state["probe_completed_epoch"]
        if epoch == previous and probe_ref == self.state["probe_ref"]:
            return self.state
        if epoch != previous + 1 or not 1 <= epoch <= 3:
            raise ValueError("Commit the next complete probe epoch, for three total epochs")
        self.state["probe_completed_epoch"] = epoch
        self.state["probe_ref"] = probe_ref
        self._save()
        return self.state

    def probe_published(self, probe_ref=None):
        if not self.state or not probe_update_due(self.state["completed_actor_outer"]):
            raise ValueError("This actor boundary has no periodic probe publication")
        if self.state["probe_completed_epoch"] != 3:
            raise ValueError("Publish after all three fitted epochs are saved")
        if probe_ref is not None and probe_ref != self.state["probe_ref"]:
            raise ValueError("Publish the same completed raw/EMA/Adam/RNG checkpoint")
        self.state["probe_pending"] = False
        self.state["probe_effective_from_outer"] = self.state["completed_actor_outer"] + 1
        self.state["phase"] = self._phase()
        self._save()
        return self.state

    def grounding_published(self, version, receipt):
        """Caller supplies the existing refresh receipt, including all4adoptions."""
        if not self.state or self.state["completed_actor_outer"] not in (50, 100, 150):
            raise ValueError("No reward refresh is due at this boundary")
        self.state["grounding_version"] = version
        self.state["grounding_receipt"] = receipt
        self.state["grounding_pending"] = False
        self.state["phase"] = self._phase()
        self._save()
        return self.state

    def next_action(self):
        if self.state is None:
            return "initialize"
        if self.state["probe_pending"]:
            return "fit_probe"
        if self.state["grounding_pending"]:
            return "refresh_grounding"
        if self.state["completed_actor_outer"] == 200:
            return "final_evaluation"
        return "next_outer"
