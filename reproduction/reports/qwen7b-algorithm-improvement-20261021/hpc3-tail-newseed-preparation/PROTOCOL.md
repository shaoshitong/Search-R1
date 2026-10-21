Prepare only: independent HPC3 tail newseed G5

Freeze: source metadata epoch1792584243,1081 sealed/138274 questions; pending241
batches/30754 questions/153770 real trajectories. This is a snapshot, not a claim
on the continuing donor. Current124194 joint-training cohort remains unchanged.

HPC3 cannot mount the donor's NAS, so no cross-cluster atomic claim is available.
The contract explicitly sets potentialOverlap=true. Each HPC3 worker has a fixed
round-robin subset and own batch flock; the two workers cannot duplicate each
other. Donor may meanwhile complete overlapping questions. Later accept exactly
one whole authentic G5 per canonicalqid, with original source/receipt retained;
never combine responses from two draws or overwrite donor outputs.

Sampling uses the same official initial7B revision a09a35458c702b33eeacc393d103063234e8bc28,
original prompts/token inputs, candidate ordinals, five response indices0..4,
seed20261021, sampling run70188b8e50fd4b45bc420ce58014f46e and original attempt.
The run_id field is retained ONLY for request-seed/hash compatibility and is
also explicitly sampling_protocol_source_run_id. A fresh execution_run_id,
HPC3 execution namespace/host/job, this implementation commit and own receipt
hashes identify the new actual execution. These are not donor original seals.
Numerical bit-exact output across engine concurrency/hardware is not claimed.

The donor's raw phase/no-hidden full-prefill adapter is vendored from the peer
repository's qwen7b_resample/worker.py; its source is preserved byte-for-byte.
The existing qualified HPC3 collector2b293 scientific runtime, BF16 interaction
contract, topk3 retrieval, temperature1/top_p1/top_k-1, strict ownEM and same-G5
continuation are reused. Actual donor batch config confirms eager TP1/512 seqs,
32768 batched tokens,0.9 utilization cap/3GiB reserve and bounded OOM adaptation.
Record Prefix20/states/full/ownEM only; no actor updates or hidden tensor cache.

Metadata-only preparation reconstructs each donor2shard input from the original
HPC3 eight-shard inputs by source ordinal, and requires exact original byte SHA.
The6656 real external/final identities are checked as qid/hash only. No evaluation
outcomes are read. Model/tokenizer/FAISS/encoder/corpus already exist remotely.
Asset metadata and prior official snapshot acceptance are pinned; CPU prepare
does not rehash GB weights or index. Real loading/runtime/GPU qualification is
pending an explicit launch GO, and no GPU effectiveness is claimed by prepare.

Prepared resource request: two independent normal/acd_u jobs, each2H100,
16CPU/128GiB/8h, TP1 generation plus separate FP16 retrieval GPU. No dependency.
The deploy helper intentionally has only stage/prepare/status and no submit
operation. Main training must start before a later explicit GPU submission GO.
