# G20 grouping consumer preparation

The G20 sampler is queued; no complete G20 training bank or GPU qualification is claimed.

The new standalone module consumes exactly20 distinct own trajectories per training question after four authentic G5 events have passed the external SHA/EM/seed acceptance. It produces four distinct G5 subsets, each with exact weight1/4. Random draws are uniform without replacement among all15,504 subsets. Class-balanced draws are uniform within their own Mixed/nonMixed category, with global count and original-question weighted class mass both exactly1:1. Infeasible class capacity fails before returning views. It never changes EM or drops/pads questions.

Combinatorial unranking avoids enumerating15,504 tuples per question. Eight CPU tests exhaustively check the complete bijection and category partitions for boundary/central K, identity/own-label mapping, exact weights, infeasibility, deterministic epoch-specific views, a common stratified train-only batch plan, and label-blind validation IDs.

Active G10 producers and trainers are unchanged. This is consumer preparation only; the full G20 data merge and real GPU consumer qualification remain required. Validation and final groups must be frozen without inspecting labels; no existing final evaluation was read.
