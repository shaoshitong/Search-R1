# QA150 saved-output CPU acceptance preparation

The outer/base/unit/config/joint and model metadata are bound in CPU_ACCEPTANCE_BINDING.json. CPU_ACCEPTANCE_CONTROLLER.py minimally parameterizes the already exercised QA100 wrapper and calls the unchanged published `qa_acceptance_cpu.verify(base,512)` with generation forbidden. It derives comparisons from this QA's saved aggregate; no correct-count, EM or F1 constant from QA100 is reused.

Execute only after normal commit/push/readback and an actual terminal QA150 unit/cgroup check. The existing pinned controller route must target h100-11. It records a distinct QA150 intent/proof, refuses an existing intent/proof, requires complete512/all8 receipts, and leaves the original outputs unchanged. The one CPU/Nice19 child loads only tokenizer and saved raw QA shards; HF weight files are stat-only, with accepted CP150 proof hashes reused. No GPU, generation, service restart or CP checkpoint acceptance is invoked. Preserve failure/intent if a command returns unknown; do not replay.

The source scorer/decoder remains QA commit2b0a9f7b809d63880dcb0e30bebc96ab11267f45 (qa_acceptance_cpu.py SHAe558437341105a644283790bffc4623c170c0dfffc997975c47fde2bb1be52e7); frozen training runtime33ef is untouched. This publication is preparation, not a CPU acceptance claim.
