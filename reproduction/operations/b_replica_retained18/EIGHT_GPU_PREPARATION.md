# Eight-card environment, waiting for the next experiment

Latest human instruction: change this thread to eight GPUs, synchronize the
reference B implementation, prepare the environment only, and do not train.
This supersedes the prior instruction to run this thread's independent B copy.
The old four-card instance is stopped with its own PID/start-tick identity;
complete joint1 and any uncommitted outer2 journals remain retained.

The source thread is `01a0fcbb-863f-7450-926d-db537f648265`. Its synchronized
publication is `c5f1a07e6a9f4ccee35f45ddf69ab4aa3d28e910`, including the B-only
CP50 fork, native prefix20/40/80 and expanded-set outcome reranking. At read time
it was preparing its CP50 continuation after stopping its old B at complete58;
A was left running. These are source-thread facts, not this copy's progress.

Available B schedule: outers51–100 use prefix20, top768 of5120;101–150 use40,
top1024;151–200 use80, top1536. Continue authentic G5, choose actual mixed groups
first, then fill by original predicted score/order to512 actor groups. Short
terminal prefixes use mask-zero padding. Existing periodic2560 completion,
rawLR5e-6, balanced whole-group resampling, mixed loss weight5 and EMA0.99 remain
the source implementation. New readout coordinates/moments expand with zeros.

The independent H10018 profile uses physical GPUs0–7, eight actor/probe ranks,
eight local FAISS shards and eight grounding replicas. Explicit global actor
micro16/local2/accumulation8 keeps mini128,20 opportunities and the original
eight-virtual-rank objective. Probe global51/52-question batches retain their
exact group mean despite6/7-question rank partitions. `environment8.sh` sets
the physical-world profile before Python imports. Default four-rank behavior
remains available for historical code/readers; no other checkout is edited.

All paths point to retained H10018 assets, no models or optimizer states pass
through the Mac. The prepared directory has no actor, probe, question-pool,
submit intent, GPU allocation or training process. It contains code, environment
settings and a blocked template only. Both submit and run reject
`preparation_only`/unassigned configurations before acquiring resources.

No reference CP50 checkpoint or four-card optimizer/RNG state is imported.
Four-to-eight FSDP optimizer resharding is not represented as a completed resume.
The next human experiment assignment determines initialization, algorithm and
budget; this preparation authorizes no future training by itself. CPU-only
environment/import/partition validation is not eight-GPU execution acceptance.
