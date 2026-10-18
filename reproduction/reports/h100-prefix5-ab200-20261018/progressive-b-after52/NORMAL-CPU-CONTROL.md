# Normal checkpoint reader for progressive B

The old AB CPU checkpoint reader asserted five policy tokens and a278-token flattened head. Future B100/150/200 states legitimately contain20/40/80 policy tokens, respectively. A remains five tokens.

The reader now derives the expected layout from the pinned progressive experiment and the **completed** outer. Checkpoint100 still has20 tokens,150 has40,200 has80. It validates raw and EMA readout dimensions, the unchanged copied block, all optimizer moments and real Adam/EMA/RNG counters. It does not convert, reset or rewrite the checkpoint.

`normal_checkpoint_control start` stages the published reader by SHA into a separate `preparation/normal-savepoint-tools/<SHA>/` directory **only for progressive B**. The CPU child imports scientific helpers from that arm's frozen repository, sets its process-local prefix to the saved stage, and has empty CUDA visibility. The frozen repository and running training process are never overwritten. A retains the original frozen reader. Existing proof/receipt/intent precedence prevents repeat launches. Before a committed checkpoint exists, status/start only reports pending.

Validation: three focused CPU tests cover exact50/51/100/101/150/151/200 boundaries and unchanged A, raw/EMA dimension rejection plus optimizer-clock preservation, and the real generated remote-control branch staging a separate reader with empty CUDA and the correct completed-stage prefix. Three existing retained-cache tests also pass. Fixtures contain no real model weights or datasets and start no native process. This is implementation validation, **not** actual future100/150/200 checkpoint loading; those points remain pending until saved.

Normal control remains nonblocking and must not become a healthy-training gate. Do not repeat accepted CP5/CP50 hashes or roll back to an older point to accommodate a tool.
