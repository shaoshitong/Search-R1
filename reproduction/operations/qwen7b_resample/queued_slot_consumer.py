"""New-stage slot entry: queue-aware readiness, unchanged model/sampler path."""
import argparse
import sys
try:
    import queued_health
except ModuleNotFoundError:
    from . import queued_health


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--slot-mode', choices=('source5', 'node10'), required=True)
    args, remainder = parser.parse_known_args()
    queued_health.install()
    if args.slot_mode == 'source5':
        import source_slot_consumer_worker as worker
        import rollover_source_slots
        rollover_source_slots.install(worker.owner_module)
    else:
        import relay_slot_consumer_worker as worker
        import queued_relay_slots
        queued_relay_slots.install(worker.relay_owner)
    sys.argv = [sys.argv[0], *remainder]
    worker.main()


if __name__ == '__main__':main()
