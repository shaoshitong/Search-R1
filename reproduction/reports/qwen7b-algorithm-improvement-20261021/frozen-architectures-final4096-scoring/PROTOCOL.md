# Future architecture final scorer deployment

Stage only. The source snapshot supports native and packed probes and validates two actual completed training seeds, their complete checkpoint/source/CPU acceptance proofs, and a common frozen epoch/state before sealed final journals can be read.

The staged template is deliberately `FUTURE_ONLY_NOT_FROZEN`, with no selected epoch/state. Import and template rejection are the only CPU execution performed during deployment; no checkpoint tensor, sealed journal, final label or score is opened. The two-GPU Slurm entry requires both an actual remote freeze JSON and its SHA256. Root alone owns freeze and submission.

Source scorer pin: cd9ca5c204950b49f34c6237259912277649c266. The original shared checkout/index are preserved. Deployment has explicit single-route direct/campus selection; failures retain actual errors and require inspection before another route is tried.
