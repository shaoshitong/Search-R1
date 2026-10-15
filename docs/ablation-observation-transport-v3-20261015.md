# Readonly observation transport V3

The existing V1/V2 observation command can raise an uncaught local `TimeoutExpired`, and OpenSSH may open a new connection when its requested task socket cannot be reused. A successful Node response still proves that Node observation; it does not by itself prove reuse of the originally requested master. These are observation-layer limitations, not evidence of a training exit.

`release/recursive_bran_a/ablation150_infra/task_transport_stdin_v3.py` is an independent copy of published V2 (`33852a1c7abf563c010d452649d4c83d63aa8353`). Its inner trusted management-to-Node program, target allowlist, identity/host-key handling, shell-quoted script arguments and CPU-only reader environment remain byte-identical. Existing V1/V2 and the running scientific source are preserved.

Use `--require-socket` with an explicit task socket. Its outer SSH command uses `ProxyCommand=false`, so an unavailable socket cannot silently fall back to a fresh connection. The default mode remains available for callers that explicitly want ordinary alias behavior.

Each invocation saves a small `.observation.json` containing the actual outer command exit, wrapper exit, reader exit when available, socket mode and timeout status. On a local timeout, the actual outer exit is unknown (`null`); wrapper124 is not reported as an actual SSH or training exit. Failed observations do not create or replace a Node payload output. Use a fresh output path for every invocation and inspect its receipt and actual return status; an old existing payload does not become fresh after a failed call.

`COMPLETE` means that the observation program returned0 and its output was saved. Training completion still requires its own run/PID/UID/start identity, native counters, actual exits and releases. V3 does not diagnose or fix the underlying intermittent network failure and does not launch, resume, stop or change the GPU experiment.

Validation is limited to four CPU-only contract cases (successful output, outer timeout, outer failure and reader failure), then an actual missing-socket negative control and actual healthy Node reads after publication. No model sampling, scientific fixtures, checkpoint reloads or additional evaluations are introduced.
