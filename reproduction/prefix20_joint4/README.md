# Four-GPU prefix20 joint actor/probe experiment

The user confirmed 200 formal actor outers on four HPC3 GPUs. In each outer,
sample 5120 candidate questions, obtain five authentic policy prefixes for each,
rank by `1 - product(p_correct) - product(1-p_correct)`, and take Top512 with
no random exploration. Complete those same 2560 trajectories, preserving their
generated tokens and behavior log probabilities. Actor learning rate and Search
G1 reward remain unchanged.

The downloaded epoch68 probe is the pretrained warmup: no extra warmup is
authorized. Its frozen official Qwen2.5-3B encoder sees the real question/context
and first20 retained policy tokens; detached layer27 post-block features at the
twenty policy positions enter copied block35 and Linear20. Question hidden
positions are not separately passed to the head. Preserve this exported
architecture and standalone safetensors weights exactly. The head is fitted
with its own-trajectory strict terminal EM BCE.

After each complete actor outer, fit the probe on only the newly selected,
completed 512×5 trajectories, once, with Adam LR1e-6, global batch256,
local batch64 across four ranks: ten optimizer updates. No historical replay,
old sigma3 loss, count classification, or group-joint BCE is included.

For strict-EM all-correct G5 groups only, normalize the original five soft
rewards using the mean and sample standard deviation of `[0, *real_rewards]`.
Return the five real advantages, with the original response mask. All-wrong and
mixed groups retain the original GRPO formula. A positive soft reward does not
establish strict correctness. Labels are recomputed at the balanced batch order.

## Immutable sources

- Code origin: `2fd5c19fa0d9c5d6c4eb514a7e9a308eb6bc6479`.
- HF dataset repo: `Sutoonq/Search-G1-prefix20-dataset-probe-epoch68-20261016-111a2d`.
- HF revision: `d794759ab9f28cf184128b227fb38f40b59fcedc`.
- Export frozen implementation: `4bc683883a569d7feb9d5da87307612b872e8f3a`.
- Pretrained head SHA256: `535c7bfee07374dde88b7d7f9e4f08ac874d46088ef961f15ea6ab7dd849a560`.
- Probe archive: 1,145,972,749 bytes, SHA256
  `83d6b834b1d3243085f74a27c7c021722e43ba3fd5004fcf41eb6202954132a5`.

## Implementation state

Acquisition mathematics, single-pass batching and strict-EM virtual-zero
advantages are implemented and CPU tested. The remote-only asset downloader
verifies official HF LFS metadata, both complete part hashes, concatenated
archive SHA, every extracted member, and fresh full reads.

The native prefix pause/selection/continuation integration and four-rank probe
worker are under construction. This directory is not yet a training launcher.
The old upstream eight-GPU frozen-actor script cannot launch this experiment.
No new joint training job has been submitted at this implementation milestone.

Never treat the artificial20-token pause as an invalid action or insert an
environment correction at that boundary. Preserve actual20/remaining-budget
sampler records; do not relabel a20-token request as a500-token request. The
pre-launch reviewer must check split boundaries, short terminal trajectories,
cross-boundary closing tags, exact row/group identity, masks and original LP.

Checkpoints, original trajectories and tensors remain remote. Local storage is
limited to code, configuration, manifests and small audit summaries. Existing
healthy experiments are not restarted or edited. Formal update201 is forbidden.
