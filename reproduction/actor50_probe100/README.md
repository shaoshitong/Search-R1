# Frozen actor50: balanced probe warmup100

The latest direct human instruction stops the formal actor run and its scheduled
monitor. Only the saved actor0050 and online probe50 are scientific inputs to
this new experiment. There is no actor optimizer, actor update, continuation of
actor200, or reuse of probe100/110. Existing assets are not deleted.

The original online50 joint metadata SHA is
`80c5b4cdbafed1c92441352cd725cc7b2819319da37f7672498f58524ec79f2c`.
The immutable probe50 payload SHA is
`771f8c5c7b1a359cf786b5353a46261d23d2999567b3c8ba1bef3e0455c16184`.
Actual actor HF file hashes and probe payload/model finite values are checked
on the remote host before the first training operation.

There are **50 cycles and 100 new epochs**, not 100 cycles. Each cycle samples
512 distinct approved training questions, generates their original five
trajectories using the frozen actor50, and obtains unchanged strict binary EM.
The same 512 groups are used for two training epochs. Across the 50 cycles the
training questions follow one seeded permutation without replacement and
exclude the exact fixed512 probe-test question IDs and normalized texts.

Whole groups are categorized by K=sum(own strict EM): all-wrong K=0,
all-correct K=5, mixed K=1..4. For each epoch the largest class defines N; all
original groups are visited and smaller classes are oversampled as complete
authentic G5 units to N each. Thus an epoch has exactly 3N groups, while a
rollout still has exactly 512 groups. No labels or synthetic groups are made.
If one genuine class is absent, the run fails explicitly instead of fabricating
data or silently changing the rollout budget.

The probe50 **raw model weights and native buffers only** initialize the new
raw head and EMA head. Adam is new at zero for this explicitly new warmup.
LR remains 5e-6, weight decay0, and the existing trajectory BCE plus stable
G5 mixed-event BCE retain weights1:1. Four physical ranks reduce the true global
whole-group mean with FP32 model/Adam and BF16 forward. Full groups remain on a
single rank per occurrence. Global batches usually contain51 groups/255
trajectories, with the final smaller batch reported exactly. No gradient or
optimizer clock is reset between cycles.

EMA uses decay0.99, the previous warmup convention, and updates once after each
actual returned Adam step. The EMA clock starts at zero with these probe50
weights. Every two epochs the same EMA head is evaluated on the original
unsampled512 training groups and the fixed512 probe test groups. The latter
are generated once by actor50 and held fixed as actual G5 events for all100
epochs. This is the original **probe** test identity manifest, not actor QA EM.
Each report includes ordinary accuracy, mixed balanced accuracy, recalls,
confusion counts, mean predicted Mixed probability and both loss terms. Train
and test use threshold0.5 and the same unchanged mixed-event formula.

The existing native Prefix5 capture, strict scorer, rollout stream contract,
BF16, context4096/action500/observation500/reserve500, and the admitted eager,
prefix-cache-off/chunked-prefill-off runtime remain in use. Test data never
participates in balancing or optimization. No independent feature encoder is
introduced.

All actor/probe tensors, checkpoints, feature mmap files, trajectories and full
predictions stay on remote `/data`. Mac and public Git contain code, config,
documentation, scalar summaries and hashes only. The runner retains a complete
raw/EMA/Adam/four-rank RNG checkpoint every two epochs and the final100 epoch
checkpoint. It owns only its new process tree and the approved four training
plus two auxiliary UUID leases. No new App scheduled automation is created.

Entry points:

* `python -m reproduction.actor50_probe100.control --action stage --handle ... --human-intent ...`
* `python -m reproduction.actor50_probe100.control --action submit --handle ...`
* `python -m reproduction.actor50_probe100.control --action observe --handle ...`

Staging, actual source qualification, idle GPU admission, actual first
rollout, first two real epochs, continuing cycles and final100 epoch acceptance
are distinct evidence boundaries. A submit receipt is not a training result.
