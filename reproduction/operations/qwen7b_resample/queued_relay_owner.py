"""New-stage CPU owner entry: queue-aware /health readiness and observation."""
try:
    import queued_health
    import retrieval_relay_owner as owner
    import rollover_terminal as terminal
    import queued_relay_slots as slots
except ModuleNotFoundError:
    from . import queued_health, retrieval_relay_owner as owner, rollover_terminal as terminal
    from . import queued_relay_slots as slots

# Import-time inspection and control-side helpers expose this generation's
# contract without mutating the original owner's module registry on import.
SLOTS = slots.specifications(owner.SLOTS)
DEFAULT_SLOTS = slots.DEFAULT_SLOTS


def __getattr__(name):return getattr(owner, name)


def verify_predecessor(pool, args):
    proof = terminal.verify_proof(args.predecessor_proof, args.predecessor_sha256, 'h100-10')
    terminal.release(terminal.core.read(proof['source_plan']))
    return dict(path=str(args.predecessor_proof), sha256=args.predecessor_sha256,
                status=proof['status'], actual_terminal_only_proof=True)


def main():
    owner.verify_predecessor = verify_predecessor
    owner.SLOTS, owner.DEFAULT_SLOTS = slots.specifications(SLOTS), DEFAULT_SLOTS
    queued_health.install()
    owner.main()

if __name__ == '__main__':
    main()
