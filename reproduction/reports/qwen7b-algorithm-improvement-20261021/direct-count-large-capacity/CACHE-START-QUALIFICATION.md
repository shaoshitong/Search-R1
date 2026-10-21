# Native frozen-E1 cache has started

Job **743978** passed the same-API golden check with the actual microbatch of 8 and BF16 prefix hidden states, without additional quantization. The check binds IDs and mask/position hashes.

The finite observer measured durable cache progress on repeated real reads. See `actual-cache-start-qualified.json` for the first two and latest observations and first/last accepted chunk receipts. Chunk acceptance follows the producer fsync and span-SHA check; this report does not repeat bulk source hashing.

The total target is 351,360 trajectories. The cache is still incomplete and head fitting has not started. CP1, replay and head-update progress remain pending. An earlier failed auxiliary CPU fixture step did not terminate the main training allocation.

The immutable v2 runtime is `805123aebf8af83a7c1c44947f39c294f7aeb624`, configuration `fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767`. No external/final scoring or new jobs were introduced.
