"""One prefix stage for selection, fitting, diagnostics and exact resume."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixSchedule:
    initial: int = 80
    later: int = 160
    switch_after_outer: int = 100
    probe_interval: int = 10
    final_outer: int = 200

    def selection_prefix(self, outer: int) -> int:
        if not 1 <= outer <= self.final_outer:
            raise ValueError("Selection outer must be in 1..200")
        return self.initial if outer <= self.switch_after_outer else self.later

    def update_due(self, completed_outer: int) -> bool:
        return 0 < completed_outer < self.final_outer and completed_outer % self.probe_interval == 0

    def update_prefix(self, completed_outer: int) -> int:
        if not self.update_due(completed_outer):
            raise ValueError("Probe updates follow completed outer 10,20,...190")
        return self.selection_prefix(completed_outer + 1)

    def update_binding(self, completed_outer: int) -> dict:
        return dict(completed_actor_outer=completed_outer,
                    rollout_actor_outer=completed_outer - 1,
                    probe_effective_from_outer=completed_outer + 1,
                    prefix=self.update_prefix(completed_outer),
                    questions=2560, trajectories_per_question=5, epochs=3)


DEFAULT_SCHEDULE = PrefixSchedule()
