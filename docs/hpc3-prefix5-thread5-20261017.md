# Independent HPC3 Prefix5/P10 run

Human scope: replace this chat's question-only periodic25 experiment with the
reference U-Search-G1 Prefix5/P10 algorithm, using four training H100 GPUs plus
four auxiliary H100 GPUs shared by the retriever and grounding services.

Source: 79c645a4a51bd0bcb0fab4ee36b8fcd634290b2f plus the candidate-schedule
snapshot recorded in reproduction/hpc3_prefix5/reference-candidate-snapshot.json.
This checkout, namespace, locks, state and Slurm launcher are independent. The
reference chat's running process and checkout were never changed.

Ordinary outers use 5120 authentic G5 prefixes, then predicted Mixed Top512 and
continue those same prefixes. Outers10/20/...190 use2560 complete G5 groups,
prioritize actual strict-EM mixed groups (probe-score fallback), and update the
probe once over12800 trajectories:50 Adam steps,LR5e-6,globalbatch256. Outer200
is ordinary and does not refit. Actor remains512G5/globalmini128/oneepoch,
20 opportunities/outer, original micro2 reduction, LR1e-6/57outer warmup,
all-correct-only virtual zero, unchanged safeguards and actual write accounting.
The native actor layer27 question+five-policy-token features feed the official
Qwen block35/flattened correctness head. Trajectory BCE and derived G5 Mixed BCE
have equal weight. No extra encoder or original question-only STD head is used.

Official Qwen2.5-3B-Instruct aa8e725 initializes actor/reference anew. Only the
retained epoch5 EMA probe model/buffers are imported, with original qualification
metadata; online probe Adam starts0. Original warmup SHA b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594;
compact model-only SHA0bf01eddd3cdaec02f91ce8a09d8c5d49d795428e634972791c99179c862d9aa.
The producer extractor accepted the compact representation. The full warmup
optimizer, raw head, cache and RNG are not transferred/imported.

Ten necessary files,359731200-byte tar,SHA73c17b3c2c91dcd3af8546d551c6d1b2c06464cf9e881841c87af92ac235a9ab,
were copied A800->HKUST2->HKUST3->HPC3. Every hop and final extracted file passed
SHA. HF source TCP443 timed out, so the faster authenticated direct remote SSH
route was used. A source-only short-lived authentication agent was destroyed
when the A800->HKUST2 transfer ended; no private key was copied, no model bytes
passed through Mac, and its proxy/network configuration was unchanged.

Target: /data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017.
Already-verified official weights, canonical train169028/dev512/final51713,
E5/wiki and the established HPC3 environment are reused read-only. Only code,
small manifests/receipts and configuration stay local. The grounding heads and
calibration300 identities are the reference run's actual original assets;
subsequent grounding fits remain0/50/100/150. QA remains5/50/100/150/200 and the
full51713 final. Source records actual Adam/skips and saves full actor/probe/RNG
states; no started process or RUNNING state is claimed as completed training.

HPC3 port changes resource binding/paths, supports four auxiliary FAISS shards,
and reuses the proven unreaped-direct-child fallback on the kernel without
pidfd. It never signals a recycled or ledger-only PID through that fallback.
CPU candidate clocks passed6 tests; the seventh initially lacked its preserved
AST-proof fixture, which is included here and rerun independently. This is not
a GPU training qualification. The real first complete iteration remains the
training-start acceptance gate.

Final logic audit (2026-10-17): deployed revision
`c578d1809b8cc587fae7b7264c560e8133bd095d` passed78 targeted CPU tests
on the actual HPC3 environment, with zero failures/errors/skips. Coverage:
whole-G5 candidate clocks and identity, unchanged-prefix continuation, native
layer27 capture positions, joint-BCE analytic values/gradients/extreme logits,
unequal-DDP-batch weighting, physical4/reference8 actor value and gradient
equivalence, strict-EM mixed selection, all-correct-only virtual-zero statistics,
actual Adam/skipped-opportunity accounting, storage boundary admission, and
shared train/QA interaction/grounding clocks. These are CPU checks, not a real
four-GPU save/load or first-outer acceptance.

Two integration defects were repaired before launch: stopped-writer recovery
now resolves only this run's owned `repo -> source/<commit>` publication alias
while checkpoint paths still reject symlinks; on a kernel without pidfd,
cleanup gives the authenticated live torchrun leader time to forward signals
and reap children. It never falls back to numeric signaling of descendants.
The latter passed an actual Linux parent/child termination test with pidfd
disabled. A stale prefix-receipt test fixture was also updated to include the
required shared interaction protocol/BF16 fields; production validation stayed
strict. An initially misspelled test-module invocation was corrected.

Actor, actor-safety, gradient-clipping, hidden capture, native probe model,
joint loss, and train/QA interaction files are byte-identical to reference
`79c645a4a51bd0bcb0fab4ee36b8fcd634290b2f`. Candidate ordinary5120/P10 pool
changes retain the14 previously sealed probe training/loading/optimizer ASTs.
The exact existing job728815 was briefly held only while publishing/testing
this fix and then released without a second submission. Online probe updates
are every10 outers, one epoch/50 Adam writes, continuing its own parameters and
Adam; this is the requested reference recipe, replacing the old P25/20epoch
recipe. Random exploration is0 in this reference configuration. No claim of
zero possible runtime defects or GPU restore correctness is made before the
actual allocation and checkpoint exist.
