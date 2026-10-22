# Viewed NQ development and fixed-train label provenance

Both pools contain NQ questions only. The fixed development2048 ownG5 labels (10,240 trajectories) and fixed training-B2048 ownG5 labels (10,240) each match their actual chosen own-EM source with zero missing identities, duplicate sources, wrong question/actor identities or label mismatches.

Development uses actual execution runs7276b1c7… andb5a4ca3f…; all32 full payloads, own-EM files and seals were freshly SHA-checked. FixedB uses actual runsbdbb550f… and70188b8e…; all492 own-EM files and seals were freshly SHA-checked. The fixedB pass binds full payload SHA references from the actual causal fixture and seals and verifies file sizes; it does not claim fresh hashes of all492 large full payloads. A separate all81920 pilot audit performs that and compares every used Prefix80 and initial prompt.

Exact question intersections with the HPC3_first9344 cohort are zero for pilot8192, viewed development2048, and fixedB2048. This does not imply correctness of the new full169028 physical-source producer. Identical protocol trajectory IDs or Prefix20 strings are never treated as complete payload identity.

The two scoped raw pass reports contain a legacy general status string; only the named completed pool was tested in each. `development-and-fixedB-source-summary.json` combines their actual tested scopes. No final4096 input, score or label was read. No model, GPU, inference, training source, or Slurm state was changed.
