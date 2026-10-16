# Final 8×TP1 terminal delta

This package is source preparation. It has not consumed a collector qualification or called a helper. Execute only after ROOT retains the actual COMPLETE run-result and explicitly authorizes the single read. It does not run an evaluator, stop a unit, load/hash model or dataset assets, or recollect training costs, variance, or curves.

`FUTURE_ACTUAL_INPUT.template.json` contains the actual existing small proof descriptors and frozen source descriptors. Its status deliberately fails the executable contract and its `root_ready_proof` is null. The future ready proof must retain the original run-result object with this exact unit, status COMPLETE, worker_exit_code 0, evaluated_count 51713. A queued action, final-metrics alone, or partial scalar score progress is insufficient. ROOT can additionally preserve the final systemd state, including exact InvocationID, CPUUsageNSec and ExecMain start/exit monotonic values before transient-unit garbage collection. Without a retained terminal value the corresponding total remains null; a live CPU observation is not a terminal total.

After ROOT authorization, create one new actual local input from the template, replacing status with `ACTUAL_COMPLETE_ROOT_AUTHORIZED_READ_READY` and `root_ready_proof` with the real `{path,bytes,sha256}` descriptor. Paths may be absolute or workspace-relative; no parent traversal is allowed. No future SHA/count/success is prefilled. For example, this local-only preparation snippet uses the actual ready file supplied by ROOT and creates, rather than overwrites, the request:

```python
import hashlib,json,pathlib
WS=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2')
package=WS/'work/tree-speed-repair-20261008/nested-final-original-costs200-source/reproduction/tree-final-fast-terminal-delta-20261016'
ready=WS/'outputs/tree-nested-branch-handoff-20261013/ACTUAL_ROOT_READY_PROOF.json'  # actual ROOT filename
raw=ready.read_bytes()
request=json.loads((package/'FUTURE_ACTUAL_INPUT.template.json').read_bytes())
request['status']='ACTUAL_COMPLETE_ROOT_AUTHORIZED_READ_READY'
request['root_ready_proof']={'path':str(ready.relative_to(WS)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
target=WS/'outputs/tree-nested-branch-handoff-20261013/FINAL200_FAST_TERMINAL_DELTA_INPUT.json'
encoded=(json.dumps(request,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
with target.open('xb') as stream: stream.write(encoded)
print(len(encoded),hashlib.sha256(encoded).hexdigest())
```

The master independently checks the COMPLETE object and every frozen input before atomically claiming `FINAL200_FAST_TERMINAL_DELTA_ATTEMPT.json` with O_EXCL. There is no new remote prepare phase. With the actual request bytes/SHA from the snippet, run once from the workspace:

```sh
env TREE_MGMT_CONTROL_PATH=/tmp/tree-mgmt-3fw9a1ab/control python3 -B work/tree-speed-repair-20261008/final_fast_terminal_delta.py collect --workspace /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2 --pins outputs/tree-nested-branch-handoff-20261013/FINAL200_FAST_TERMINAL_DELTA_INPUT.json --pins-bytes ACTUAL_BYTES --pins-sha256 ACTUAL_SHA256
```

One helper call uses the original retained18 helper and h100-10, never a second evaluator or cleanup action. It performs two internal terminal frames two seconds apart and stable metadata readbacks. Target command ≤32768 bytes, management command ≤65536, argv ≤80000; lossless stdlib lzma carries the bounded program/input. Complete scalar-only stdout ≤4 MiB and stderr ≤32768 bytes; exact framed response is durable before local product splitting/publication. On an unknown result inspect the existing once marker and response. A successful response can be rendered again only locally:

```sh
python3 -B work/tree-speed-repair-20261008/final_fast_terminal_delta.py recover --workspace /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2 --pins outputs/tree-nested-branch-handoff-20261013/FINAL200_FAST_TERMINAL_DELTA_INPUT.json --pins-bytes ACTUAL_BYTES --pins-sha256 ACTUAL_SHA256 --response-bytes ACTUAL_RESPONSE_BYTES --response-sha256 ACTUAL_RESPONSE_SHA256
```

The remote read is limited to exact own runtime/source metadata, 809 committed score chunks plus their done receipts, eight provenance records, the own resource JSONL (≤256 MiB streamed on the node), the saved two-CPU-unit cleanup receipt, and three retained original778 final-only metadata JSON files. Each score chunk is ≤262144 bytes and normal small JSON/source files ≤1 MiB. Original raw scores and resource rows remain remote; only scalar sums, coverage, source paths/stat/SHA and chunk descriptors return. No raw rollout, token arrays, tensor, PT, HF, dataset, FAISS, or environment assets are transferred or hashed.

Score validation requires 809 chunks, 51713 unique contiguous indices 0..51712, exact seven-source counts, eight equal provenance records and final-metrics/done SHA agreement. Stored per-question scalar scores are independently summed with math.fsum and compared with final-metrics. Stored question SHA format is checked; dataset content is not reloaded by this collector. The original merger separately checked stored question hashes against the dataset. Greedy batching/kernel changes are not claimed bit-identical to the stopped evaluator.

Batch wall sums overlap across ranks; batch process CPU is a subset of service CPU and is not added back. Whole-card utilization/memory peaks and own/foreign memory remain distinct. Eight times systemd execution wall is only shared physical capacity, not own GPU active time or a bill. Unrecorded scoring/HTTP/energy/own GPU active/billed costs stay null, while recorded counters retain coverage and known sums. Invalid/partial resource lines and actual sample gaps are reported.

Cleanup requires the original fast unit's terminal/PID0 and empty own recursive cgroup, released own cooperative lease FDs and no own listener/process residue. Foreign GPU users, foreign lease holders, and a subsequently reused port are recorded without operation and do not require global idle. The two CPU audit units are not requeried or stopped: their exact saved ROOT cleanup receipt preserves successful pre-stop invocations/counters, two inactive/PID0/empty-cgroup observations and post-GC unavailable counters as null. Old3360 is not included. Original778 FAILED is retained as a user-authorized final-only stop after training200. Its full-segment CPU1116210.663921s includes training/dev/partial-final; only the original final control's directly recorded counters/wall isolate a final subset. Final-only service CPU/wall remain null when not separable.

Outputs are small immutable metadata/field parts, each part ≤100000 bytes, mirrored to the existing two destinations with two real byte readbacks. Neither this source preparation nor collector success asserts goal completion. Source1ed remains the already executed 7/7; this collector's actual calls and observer CPU/wall are separate, and prior failed observers/collectors remain separate historical evidence.
