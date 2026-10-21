"""Own-process-only capacity declaration; unchanged supplementary controller.

The original finite GPU4 anchor remains visible to owner busy_slots. The
controller owns only its new model children and borrows the live retriever.
"""
import source_supplementary_worker as original

SLOT = 'gpu4-tp1'
SPECIFICATIONS = {SLOT: dict(indices=[1, 3, 4, 6], tp=4, seqs=32, tokens=8192, blocks=4096, reserve=.5)}


def admit(pool, args):
    previous = original.SPECIFICATIONS
    try:
        original.SPECIFICATIONS = SPECIFICATIONS
        return original.admit(pool, args)
    finally:
        original.SPECIFICATIONS = previous


def main():
    import queued_health
    queued_health.install()
    consumer = original.consumer
    saved = consumer.admit, consumer.borrowed_relay
    try:
        consumer.admit = admit
        consumer.borrowed_relay = original.slot_worker.borrowed_relay
        consumer.main()
    finally:
        consumer.admit, consumer.borrowed_relay = saved


if __name__ == '__main__': main()
