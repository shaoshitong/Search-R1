# Reference B: independent HPC3 environment, preparation only

Human instruction2026-10-19: inspect chat01a0fcbc-0b46-7502-8b66-d8b21a0e032d
(title `U-验证 Tree-GRPO 效果`), prepare its code/data/logic/probe environment;
**do not run training**. The explicit initialization answer is official initial
Qwen2.5-3B-Instruct actor plus the same epoch5 EMA probe, not its CP50/latest
actor or online optimizer.

Observed live reference config SHA:
`0d4ab4c0bba08f8244afa5d3fce0bcf4ab97450b8fa93c218340da416517ae88`.
Actual immutable reference runtime:
`bc8b16728b689ccbb97457e06f2ef0a65cb39fa7`.
This branch starts at that exact commit; preparation adds only this operations
directory. The reference's currently running H1005 instance is untouched.

Target:
`/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019`.
Its own repo, metadata, overlay, FAISS GPU libraries, caches and temporary
directories are independent. Already qualified shared model/data/retrieval files
and immutable installed package directories are reused read-only.

## Scientific behavior being prepared

- B arm: class-balanced current-outer whole-G5 fitting; raw Adam LR5e-6,
  trajectory BCE plus group-loss weight5, selection EMA0.99.
- Ordinary actor data512questions: online probe fits10real writes per rank per
  ordinary outer. P10 uses2560questions and50writes, one epoch. This differs
  from this chat's finished P10-only/three-epoch experiment.200does not fit.
- Ten actual native prefixes/question are reduced to five by deterministic
  token positional/bigram farthest-first diversity. That step uses neither
  probe outputs nor EM. It preserves original five trajectory identities.
- P20/P40/P80 stages continue predicted Top768/1024/1536 of5120, respectively,
  then strict actual-Mixed-first512 with existing-score fills when needed.
  From official initialization, Prefix5applies through50 and subsequent
  prefix stages start51/101/151. The reference itself acquired this later
  protocol at a CP50fork; fresh0is a new planned run, not its historical resume.
- Official revisionaa8e72537993ba99e69dfaafa59ed015b17504d1;
  commonmanifest0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24.
  Original train169028/dev512/final51713, BF16 train/QA and FP32master/Adam,
  micro2mathematical objective, actor512G5/min128/20opportunities and original
  strict scorer remain in the reference source.
- Eight-rank reference topology is recorded as a plan, not allocated GPUs.
  GPU UUIDs, HPC3 execution/namespace binding and any actual scheduling remain
  future execution work after user authorization. No Slurm script is submitted.

## Weight and dependency equivalence

H100reference warmup fileSHA06ffd8c21b6208ff5a0beb11cded62da7b9551c3e1357a1d7321e69eb06d994e
differs from retainedHPC3serializationSHA0bf01eddd3cdaec02f91ce8a09d8c5d49d795428e634972791c99179c862d9aa.
CPU extraction confirmed all14EMA learned tensors and both nonpersistent
buffers match exactly by dtype/shape/full byteSHA. Complete epoch5/run/source
binding is validated with the original extraction routine. No warmup optimizer
or RNG is imported and no online state is constructed in preparation.

Both environments report torch2.4.0, transformers4.46.3, vLLM0.6.3,
datasets3.2.0, safetensors0.8.0 and numpy1.26.4. The private CPU-check wrapper
uses these existing HPC3 packages and a private copy of the existing sklearn
overlay/FAISS GPU library. No package build or external package download is
needed when the actual import verification passes.

## Verification boundary

`prepare.py` only creates private configuration/directories and copies remote
support libraries. `inspect.py` checks all deployed source SHA, real package
imports, eight-rank mathematical plans, balancing/diversity, small data SHA and
exact EMA/buffer identity. It calls the reference's existing preparation-only
training guard, never a launch/submit entry point. CUDA stays uninitialized;
no actor forward, rollout, optimizer step, checkpoint resume or GPU/DDP test
occurs. Large shared model/index/corpus files are not copied or rehashed here;
they retain prior accepted provenance and fresh path/stat presence checks.

Run CPU verification only:

```bash
/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019/env/bin/python-cpu-check \
  -m reproduction.operations.reference_b_hpc3_prepared.inspect \
  --base /data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019
```

Independent inspection receipts and final deployment report are kept in the
target's `preparation/` and this chat's `outputs/reference-b-deployment-20261019/`.
Preparation is not a claim of GPU training readiness or a full-state resume.
