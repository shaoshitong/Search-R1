The 2026-10-15 human instruction replaces the weak two-epoch warmup with a fresh
ten-epoch fit on the same authenticated 5120 frozen-official-teacher questions
(five actual strict-EM trajectories per question). No held-out warmup split is
used. Three blocks are freshly copied from the official final decoder layer;
neither the old trained probe nor its Adam state is inherited. The actor and
fixed reference also restart from official weights for a new 200-outer run.

Warmup uses logical batch256, physical micro4 and LR3e-4: 20 Adam updates per
epoch, 200 total. Online probe fitting remains two epochs per completed actor
outer. The existing std3 CE + normalized sigma MSE +0.1 count6 CE is unchanged.

The remote scalar curve records epoch0, each optimizer update, and epochs1–10.
Every epoch evaluates all5120 training questions using Pmixed=P(1<=K<=4), with
threshold0.5, confusion counts, accuracy, precision, recall, F1, BCE and Brier.
The majority-class baseline is explicit. Both Pmixed and predicted-sigma Top448
mixed hit rates are recorded against the full training-pool random expectation.
These are in-sample metrics, not held-out validation or a promised future yield.

The real first actor rollout must separately report pre-generation/prefit
Top448 mixed counts versus random64, with the new warmup head version. Actor
iterations or finite gradients alone do not establish probe usefulness.

The same teacher trajectories/features may be reused: only the metrics/fitting
implementation hash of qprobe_warmup.py is explicitly mapped to the cached
teacher identity. Model, data, prompt, sampling, retrieval and scoring identities
still compare exactly. No new teacher rollouts are claimed when cache is reused.

Superseded actor checkpoints from jobs3399/3400 and their incomplete CP5 backup
are deleted only under the user's explicit deletion request, after owned readers
and writers stop. Official assets, teacher labels/features, source and scalar
history remain. Deleted checkpoints cannot be claimed recoverable or evaluated.
