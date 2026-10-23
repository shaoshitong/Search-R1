"""Prepare or submit the authorized lean continuation using existing recovery.

Call submit only after the parent's measured speed/numerical decision and its
authorized boundary stop. This command never stops a producer. It selects the
latest sealed full source checkpoint after the old producer is terminal, uses
the existing full-state receipt publisher, then invokes the normal native4
launcher. No extra startup benchmark, numerical gate, or watcher is added.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from reproduction.operations.easy7b_20261021 import fa3_transition as transition

SOURCE_BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022'
SOURCE_JOB = '747219'
SOURCE_COMMIT = 'b76d7a05245a5eb7adc75385e2db803c67acce43'
TARGET_BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-lean-r1-20261022'


@contextmanager
def source_binding(*, source_base, source_job, source_commit, authorization, minimum_outer=50,
                   infra_profile='fa3-lean-v1'):
    """Parameterize the historical helper only inside this private process."""
    source = transition.control.own_base(source_base)
    source_run = json.loads((source / 'logs/run-manifest.json').read_text())
    old = {name: getattr(transition, name) for name in
           ('SOURCE_BASE', 'SOURCE_JOB', 'SOURCE_COMMIT', 'RUN_ID', 'PROFILE',
            'MINIMUM_OUTER', 'AUTHORIZATION', '_consumers')}

    def consumers(base, commit, job=None):
        from search_g1.easy2560_rolling_checkpoint import latest_checkpoint
        latest = latest_checkpoint(source, verify=False)
        completed = latest['step'] if latest is not None else 0
        remaining = [step for step in (5, 50, 100, 150, 200) if step > completed]
        return dict(base=str(base), source_commit=commit, source_job=job,
            run_id=source_run['run_id'], source_completed_outer_observed=completed,
            prior_archive_dependencies='Retain accepted source and ancestor milestone/core/post-callback/dependency archives',
            remaining_milestones=remaining,
            core_controller_first_step=remaining[0] if remaining else None,
            core_controller='Bind a new independent controller to target base/commit/job; do not edit source controller',
            post_callback_companion='Bind a new independent target companion with the preserved run_id',
            final_evaluation='Native51713 + seven source metrics after200; final native4 actual restore separately required',
            final_update=200, update201_allowed=False, archive_processes_started_here=False)

    transition.SOURCE_BASE = source
    transition.SOURCE_JOB = str(source_job)
    transition.SOURCE_COMMIT = source_commit
    transition.RUN_ID = source_run['run_id']
    transition.PROFILE = infra_profile
    transition.MINIMUM_OUTER = minimum_outer
    transition.AUTHORIZATION = authorization
    transition._consumers = consumers
    try:
        yield transition
    finally:
        for name, value in old.items():
            setattr(transition, name, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'submit'))
    parser.add_argument('--base', default=TARGET_BASE)
    parser.add_argument('--commit', required=True, help='Published immutable target runtime commit')
    parser.add_argument('--source-base', default=SOURCE_BASE)
    parser.add_argument('--source-job', default=SOURCE_JOB)
    parser.add_argument('--source-commit', default=SOURCE_COMMIT)
    parser.add_argument('--minimum-outer', type=int, default=50)
    from search_g1.easy7b_infra import LEAN_PROFILES
    parser.add_argument('--infra-profile', choices=LEAN_PROFILES, default='fa3-lean-v1')
    parser.add_argument('--authorization-json', required=True,
                        help='Existing current user authorization and measured qualification decision record')
    args = parser.parse_args(argv)
    authorization = json.loads(Path(args.authorization_json).read_text())
    with source_binding(source_base=args.source_base, source_job=args.source_job,
                        source_commit=args.source_commit, authorization=authorization,
                        minimum_outer=args.minimum_outer, infra_profile=args.infra_profile) as bound:
        result = (bound.prepare if args.mode == 'prepare' else bound.submit)(args.base, args.commit)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
