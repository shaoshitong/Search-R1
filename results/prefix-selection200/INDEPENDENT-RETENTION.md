# Independent remote joint-state retention

Training continues on h100-8. The CPU-only copier reads the existing h100-8 NFS export directly from h100-10 and writes a separate local ext4 copy. Model/optimizer/tensor/rollout bytes never pass through Mac. Both copy and destination full SHA readback use20MiB/s;128GiB ordinary free capacity remains reserved for other use.

Only complete5/50/100/150/200 points with an independently authenticated full SHA/CPU state proof are admitted. The checkpoint's saved pool boundary, immutable probe milestone and actor state are copied; mutable live latest pointers are excluded. Original receipts keep their original paths, accompanied by RESTORE_MAPPING.json for a future explicit recovery adapter. Original retained tokens/selection/outcomes/metrics are included; consumed feature caches are unnecessary. External input manifests and source fingerprints remain explicit.

INTENT/RECEIPT/PROGRESS/RESULT are separate. A pre-existing namespace is queried, never replayed. PASS requires every actual source file and independent destination file to be fully read and match, then an atomic final-directory rename. This copy does not perform model reload or GPU work and never claims all experiment assets/logs are already backed up.

Current source5 is the independently validated resume2 joint5; immutable actor/probe Adam100/50 per rank, scheduler5 and cursor25600/next6. The current execution, its future milestones, all final outputs and external runtime assets remain distinct retention work.

Validation:3 small CPU tests cover path rejection, real copy plus independent destination digest, and contradictory source proof rejection. Source metadata admission runs on the destination with no model/GPU reads.
