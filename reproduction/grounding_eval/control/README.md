# OPSD 3304 iter200 asset entrypoints

The asset worker is `../asset_fetch.py`, immutable launched commit `84508362efa647d8ebed8907f9b65c5cb4cc773f`. Three CPU unit tests cover restart/resume, independent final verification, source mismatch preservation, secret exclusion, and selection limits.

Run controller scripts from `/Users/shaoshitong/Documents/Codex/2026-10-03/xia`; these reuse existing `control.py` routes. The public job and source identity receipts are backed up here. No credential is stored in these files. The existing task token is read on mgmt-host only, relayed as control metadata in RAM, then supplied to the target worker stdin; curl reads its header from an anonymous pipe. Models remain on HF/A800.

- `hf_private_metadata_probe.py`: authenticated, private repository metadata only.
- `probe_opsd_hf_a800_resolve.py`: bounded 8 MiB remote-RAM probe. It uses a per-request public CloudFront address, preserving hostname TLS verification; never edits global DNS/hosts/proxy/SSH.
- `launch_opsd_asset_a800.py`: exact published worker deployment, immutable source-bound job, and one task-scoped asset process. Recheck an existing PID/command/lock before recovering; do not duplicate a writer.
- `check_opsd_asset_a800.py`: bounded sanitized progress/PID/files receipt, never prints credentials.

Target: `/data/shaoshitong/grounding-eval-20261019/assets/opsd3304-iter200`. Selected 12 HF inference files only, 13,604,379,705 bytes. Training state, optimizer state and environment are excluded. `complete.json` is published only after every target file matches the original retained18 source SHA, is fsynced, and passes a second complete readback. No GPU/Slurm launch occurs here. Partial files are retained. Original source assets are never deleted.
