"""Fresh guarded retrieval owner with the explicit eight-slot source manifest."""
try:
    import retrieval_remote_guard_owner as guard
    import rollover_source_slots as slots
except ModuleNotFoundError:
    from . import retrieval_remote_guard_owner as guard
    from . import rollover_source_slots as slots


def main():
    slots.install(guard.owner)
    guard.main()


if __name__ == '__main__':main()
