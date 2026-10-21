LOCAL-ONLY HANDOFF — commands below are FUTURE OPERATOR ACTIONS, NOT EXECUTED.
Prepared from published controls plus the new isolated terminal seam pending
root review/publication. No SSH/query/stop/stage/submit/test-GPU/Git action was
performed for this handoff. Current healthy source5 and current primary cohort are
untouched. Snapshot time is the root's current task context, not a fresh probe.

Objective and timing
--------------------
Continue RUN70188b8e50fd4b45bc420ce58014f46e, seed20261021, same169028q/G5
bank beyond old deadlines. Around09:28 CST is an approximate new-claim cutoff,
not a guaranteed stop time: actual loop reserves use observed longest batches.
The GPU3 actual model deadline is09:48:54 CST, root nominal10:48:54; root can
return earlier. Capture all live identities WELL BEFORE those cutoffs (e.g.
08:45–09:00 once all replacement models have resident children). Do not wait
until the parent disappears, then attempt nsenter on the dead parent.

Definitions (operator fills measured hashes; do not invent them)
--------------------------------------------------------------
REPO=/Users/shaoshitong/Documents/Search-G1/qwen7b-h100-assets-20261020
ROOT=/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control
B=/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020
P=/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020
M=reproduction.operations.qwen7b_resample
REF=refs/heads/codex/qwen7b-h100-assets-20261020
PUB=ACTUAL_FULL_SHA_AFTER_ROOT_PUBLISHES_THE_NEW_SEAM
CPU_READY=$B/preparation/legacy-relay10-independent-v1/owner-ready.json
CPU_SHA=ACTUAL_IMMUTABLE_CPU_READY_SHA
PRIMARY_INSTANCES=gpu1-tp1=v2,gpu2-tp1=v3,gpu3-tp1=v2,gpu5-tp1=v3,tp4-0-4-6-7=v3
OLD4=$P/preparation/node10-native-failures-20261021-v1/complete.json
OLD4_SHA=2c3a7de77b48c08435bed4df6e045be5a44b39bfc276249925962e5848b6613d
PLAN10=$B/preparation/cohort-rollover-v1-h100-10/plan.json
PLAN5=$B/preparation/cohort-rollover-v1-h100-5/plan.json
DONE10=$P/preparation/cohort-rollover-v1-h100-10/complete.json
DONE5=$P/preparation/cohort-rollover-v1-h100-5/complete.json
PLAN10_SHA/PLAN5_SHA/DONE10_SHA/DONE5_SHA are each actual returned/readback SHA.
All commands below run from REPO using python3 -m; shell variables are shown
for brevity. They are not populated by this document and have not been run.

Current complete dependency inventory
-------------------------------------
node10 replacement cohort: CPU unit sst-q7b-g5-cpu-relay10-v1-70188b8e50fd.service
owns only the localhost relay (last root-reported CPU PID2882823). Five explicit
mixed-instance units borrow it:
sst-q7b-g5-primary10-gpu1-tp1-v2-70188b8e50fd.service
sst-q7b-g5-primary10-gpu2-tp1-v3-70188b8e50fd.service
sst-q7b-g5-primary10-gpu3-tp1-v2-70188b8e50fd.service
sst-q7b-g5-primary10-gpu5-tp1-v3-70188b8e50fd.service
sst-q7b-g5-primary10-tp4-0-4-6-7-v3-70188b8e50fd.service
The mapping above is the planned next live cohort, not evidence that a v3 unit
has started. PIDs/births MUST be captured from each selected instance's actual
allocation/processes. No v3 PID is prefilled. CPU capture must agree with its
real owner-ready and allocation. The failed v1 PID list and retired waiting-v2
instances are not replacement identities. A waiting controller is not proof
of initialized sampling; retain its genuine state and terminal outcome.

source5 eight model units (last local small actual read, rebind births live):
sst-q7b-g5-source5-extra-gpu1-v3-70188b8e50fd.service      PID2194351
sst-q7b-g5-source5-extra-gpu2-v3-70188b8e50fd.service      PID2193217
sst-q7b-g5-source5-tp1-replacement-gpu3-v1-70188b8e50fd.service PID2364335
sst-q7b-g5-source5-tp1-replacement-gpu4-v1-70188b8e50fd.service PID2271442
sst-q7b-g5-source5-tp1-replacement-gpu6-v1-70188b8e50fd.service PID2272406
sst-q7b-g5-source5-tp1-replacement-gpu7-v1-70188b8e50fd.service PID2272887
sst-q7b-g5-source5-extra-retrieval-v1-70188b8e50fd.service  PID2013773
sst-q7b-g5-sharedvram-v6-70188b8e50fd.service             PID1490214/birth6807965
The last unit owns BOTH its TP4 sampler and retriever1490465/birth6808057,
private mnt/net namespaces and all8 source5 primary UUID leases.
Also include source5 CPU upstream sst-q7b-retrieval-relay5-v1-70188b8e50fd.service
(last actual PID1654561), which depends on that retriever/namespace.

Already terminal histories remain distinct: old V5/parallel/TP1gpu1/gpu2
failed4 proof above; primaryv1 pre-model failure proof at
$P/preparation/node10-primary-v1-pre-model-terminal-20261021/complete.json;
retired source extra-model TP4 proof at
$B/preparation/source5-model-tp4-to-tp1-v1/complete.json.
Do not put these already-failed units into a fictitious currently-active plan,
or rename their failures as controlled stops.

1. Publish/stage only the new collector seam, then capture both live plans
-------------------------------------------------------------------------
Root must first commit/push/full-SHA verify the NEW files. Until that happens,
PUB above is intentionally unresolved; current017660... does not contain them.
The first actual capture10 failed before writing plan/intent because its
private /mnt/data/shaoshitong view is retained assets, not the physical NAS10
data tree. The NEW rollover_paths resolver now reads the BASE alias first and
falls back to the physical spelling only when alias is absent. Proof records
and original inode/SHA bindings still use physical NAS10 paths; SHA mismatch
never triggers fallback to another file. The path fix creates new immutable
source identities under the same generation root. Preserve the already staged
old source directories; do not overwrite them or replay any existing intent.

python3 -m $M.rollover_terminal_control stage --node h100-10 --generation rollover-v1 --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.rollover_terminal_control stage --node h100-5 --generation rollover-v1 --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.rollover_terminal_control capture --node h100-10 --generation rollover-v1 --primary-instances "$PRIMARY_INSTANCES" --cpu-ready "$CPU_READY" --cpu-ready-sha256 "$CPU_SHA" --failed-proof "$OLD4" --failed-proof-sha256 "$OLD4_SHA" --published-commit "$PUB" --published-ref "$REF"

Record actual PLAN10_SHA. This captures the five explicit mixed-instance units and CPU
owner as live, original MainPID/birth/cgroup members, exact allocations and
metadata roots. It never discovers/stops arbitrary matching units. Capture
after constructors/children are resident and all five selected units actually
exist; it must not silently fall back from v3 to the older same-slot v2 unit.
If an expected unit has already
failed, do not fill a birth from a guess or replay an old capture: retain its
real terminal evidence and have root review a truthful adjusted plan.

python3 -m $M.rollover_terminal_control capture --node h100-5 --generation rollover-v1 --remote-node10-plan "$PLAN10" --remote-node10-plan-sha256 "$PLAN10_SHA" --published-commit "$PUB" --published-ref "$REF"

Record actual PLAN5_SHA. It captures all8 source model units plus upstream5,
root/retriever births and actual old mnt/net inode, while parent1490214 still
exists. It references the immutable node10 PLAN, so both can be captured before
node10 drains; source5 proof later binds actual node10 COMPLETE+SHA.

2. Start bounded terminal-only observers before touching any predecessor
------------------------------------------------------------------------
python3 -m $M.rollover_terminal_control collect --node h100-10 --generation rollover-v1 --plan-sha256 "$PLAN10_SHA" --watch-seconds 7200 --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.rollover_terminal_control collect --node h100-5 --generation rollover-v1 --plan-sha256 "$PLAN5_SHA" --watch-seconds 7200 --published-commit "$PUB" --published-ref "$REF"

These start separate bounded own units, not agents and not stoppers. Source5
collector enters1490214's mount AND net namespaces while alive, in its own
cgroup. It survives root exit and retains NAS10 visibility. node10 collector
creates its own retained mount view. Both keep collector.log on shared NAS10.
One immutable collector intent per generation; unknown start => observe same
unit/proof, never blind replay. Status reads shared artifacts via physicalNAS10,
so it still works after root1490214 dies; it is not a collector-process heartbeat.

3. Drain node10 models through their CPU owner, then require true terminal
------------------------------------------------------------------------
python3 -m $M.legacy_relay_control drain --instance v1 --topology tp4

This writes bound drain-owner.json. The selected v2/v3 stage samplers use the drain
wrapper: finish accepted prefix+full and seal before next claim, release their
own primary UUID AND slot-live leases only after cleaning own sampler groups.
CPU owner waits all5 finite live leases, then stops ONLY its listener child.
It does not stop any source5/model unit. A readiness/transport failure is a real
failure; do not rewrite it as a successful drain. Source5 stays running here.

python3 -m $M.rollover_terminal_control status --node h100-10 --generation rollover-v1

Require actual DONE10/its SHA, not merely an inactive CPU PID: all5 selected primary
and CPU units MainPID0; their original births/cgroups/GPU contexts gone; local
28831 bind free; two actual observations>=3s. Proof status is NEW
OWN_BOUND_COHORT_TERMINAL_RELEASED_V1 with controlled_stop=false, because the
collector only OBSERVED terminal states. Old failed4 reference keeps its actual
OWN_NODE10_NATIVE_FAILURES_ORIGINAL_SHA_RELEASE_PASS status. If a model/CPU
remains active, keep source5 index alive; do not start a new owner or free a
primary lock file by deleting it.

4. Only after DONE10, retire source dependents and root LAST
----------------------------------------------------------
Existing source frozen loops have no graceful drain marker. Prefer their
natural bounded returns, observing complete/sealed batches. If root explicitly
chooses an earlier controlled retirement, the only permitted stop order is
the exact PLAN5 units: seven additional model units first, upstream5 next,
and sharedvram-v6 LAST. A snapshot full640 can race the next claim; preserve
the resulting true partial, never promise zero work loss.

For a deliberate future retirement, an operator may issue systemctl stop
ONLY each exact own unit listed in PLAN5 after rechecking its recorded birth/
MainPID. No broad PID matching, global mounts, foreign job stop or queue change.
The terminal collector itself has NO stop action. Do not use the old
predecessor_generation_terminal.py --execute with a forged V5 remote proof:
its fixed old consumer contract cannot represent this replacement cohort.

After seven extra model units are actually terminal, and DONE10 already true:
systemctl stop sst-q7b-retrieval-relay5-v1-70188b8e50fd.service
Only after upstream is actually terminal and PLAN5's exact root identity is
still the intended process, stop (or await natural return of) root LAST:
systemctl stop sst-q7b-g5-sharedvram-v6-70188b8e50fd.service

These are future exact-node5 commands, NOT executed by preparing this file.
Keep any actual manual-stop intent/receipts separately; the collector does not
invent controlled-stop history. If a unit naturally fails, record that failure.

python3 -m $M.rollover_terminal_control status --node h100-5 --generation rollover-v1

Require actual DONE5/SHA: all8 models+upstream terminal, original root/retriever
births gone, cgroups empty, own GPU contexts absent twice, all original JSON/
logs/bootstrap inode hashes stable. Source5 proof binds actual DONE10 and its
captured-plan fingerprint. It NEVER queries a node10 PID in source5 /proc.

5. Fresh guarded24h source may now arbitrate all8 primary UUID leases
-------------------------------------------------------------------
Prepared source slot set is finite and now explicitly contains eight:
base-tp4,retrieval-tp4,gpu1-tp1,gpu2-tp1,gpu3-tp1,gpu4-tp1,gpu6-tp1,gpu7-tp1.
Only the NEW rollover owner/slot entry registry declares GPU3/4/6/7 TP1 initial
caps32seq/8192batch-tokens/4096KV-blocks/1GiB reserve, matching the measured
current4/6/7 capacity and explicitly extending GPU3. Base/retrieval TP4 and GPU1/2
specifications remain unchanged; the default manifest explicitly declares all8
slots, not the old4. The old owner/worker sources
and immutable stages are not changed. The new retrieval-only owner has no old
V6 resident model; old V6 TP4 and GPU3 TP1 must first truly release in PLAN5.
This is a finite capacity declaration, NOT a CPU claim of GPU fit. The original
4096 context/science remains fixed; only actual constructor profiles and typed
OOM fresh-process retries may reduce concurrent seq/cache capacity.

SOURCE_SLOTS=base-tp4,retrieval-tp4,gpu1-tp1,gpu2-tp1,gpu3-tp1,gpu4-tp1,gpu6-tp1,gpu7-tp1
python3 -m $M.guarded_rollover_control stage --generation owner-v2 --enabled-slots "$SOURCE_SLOTS" --intent-id guarded-owner-v2-stage --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.guarded_rollover_control submit --generation owner-v2 --enabled-slots "$SOURCE_SLOTS" --predecessor-proof "$DONE5" --predecessor-sha256 "$DONE5_SHA" --intent-id guarded-owner-v2-submit --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.guarded_rollover_control status --generation owner-v2 --enabled-slots "$SOURCE_SLOTS"

This NEW consumer seam validates both real cohort proofs, then rechecks only
local source5 release. shared.admit still arbitrates actual primary UUID locks.
Pending3967/3976 are unchanged; a queued task may legitimately acquire a lock
first. Do not cancel/hold/reprioritize it or delete a lock to make this pass.
Admission/start failure or unknown outcome never permits replaying the same
intent; inspect actual status. A repaired owner retry needs a fresh generation.
Owner unit:sst-q7b-g5-guard-rollover-owner-v2-70188b8e50fd.service.
Owner directory:$B/preparation/retrieval-generation-guard-rollover-owner-v2.
Wait for the real fullFP16 index READY and returned owner-ready path/SHA. A
running unit or script publication alone is not successful index continuation.

6. Reconnect independent relay owners; then start finite model slots
------------------------------------------------------------------
Use NEW queued controls so both CPU health/readiness and model readiness use
60s, matching the accepted59s relay budget. Do NOT relaunch legacy old-V6
primary controls against the new source: those intentionally pin1490214.
The NEW node10 registry preserves the measured current5 topology and adds only
GPU7TP1:6 finite child slots covering all8 physical cards. The single TP4 is
[0,4,6,7],8seq/4096tokens/512KVcap/.125GiB reserve (current productive actual469
KVblocks). All five TP1 groups declare64seq/8192tokens/4096KVcap/1GiB reserve.
The old two contiguous TP4 groups [0..3]/[4..7] are excluded; they would add
weights/cache to current productive TP1 cards without a measured free window.
The downstream owns all8 primary UUID leases. Each overlapping finite slot
proves those locks and holds its own child lease. Control, CPU owner and model
entry all share queued_relay_slots.py; no old owner/worker is edited.
GPU7 is the sole additional model declaration based on retained24005MiB free.
The declaration does not prove future fit or performance: actual profile and
eligible typedOOM retry decide capacity with original4096/seed/G5 unchanged.

NODE10_SLOTS=tp4-0-4-6-7,gpu1-tp1,gpu2-tp1,gpu3-tp1,gpu5-tp1,gpu7-tp1

For upstream then downstream, stage their exact bundles before submit:
python3 -m $M.queued_generation_control upstream stage --generation owner-v2 --enabled-slots "$NODE10_SLOTS" --intent-id queued-up-v2-stage --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.queued_generation_control upstream submit --generation owner-v2 --enabled-slots "$NODE10_SLOTS" --source-owner-ready "$OWNER_READY" --source-owner-ready-sha256 "$OWNER_SHA" --listen-host "$ACTUAL_PRIVATE_SOURCE5_IP" --intent-id queued-up-v2-submit --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.queued_generation_control downstream stage --generation owner-v2 --enabled-slots "$NODE10_SLOTS" --intent-id queued-down-v2-stage --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.queued_generation_control downstream submit --generation owner-v2 --enabled-slots "$NODE10_SLOTS" --source-owner-ready "$OWNER_READY" --source-owner-ready-sha256 "$OWNER_SHA" --upstream-owner-ready "$UP_READY" --upstream-owner-ready-sha256 "$UP_SHA" --predecessor-proof "$DONE10" --predecessor-sha256 "$DONE10_SHA" --intent-id queued-down-v2-submit --published-commit "$PUB" --published-ref "$REF"

The future upstream/downstream chain holds the new source guard dependency
lease; downstream owns node10's8 primary leases and finite child slots. New
downstream defaults to the measured TP4 slot0/4/6/7 plus TP1 slots1/2/3/5/7,
all explicitly named above at stage/submit, with actual GPU profiling later.
This finite6 topology differs from the retired legacy5-model manifest; do not
silently substitute either manifest. Tokens are newly generated remotely for
this generation, never copied to Mac; old tokens/proofs remain preserved.

For each declared source SLOT (stage, then submit), e.g.:
python3 -m $M.guarded_rollover_slot_control stage --generation owner-v2 --slot gpu2-tp1 --instance r1 --intent-id source-gpu2-r1-stage --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.guarded_rollover_slot_control submit --generation owner-v2 --slot gpu2-tp1 --instance r1 --owner-ready "$OWNER_READY" --owner-ready-sha256 "$OWNER_SHA" --intent-id source-gpu2-r1-submit --published-commit "$PUB" --published-ref "$REF"
For GPU3, use the same two commands with --slot gpu3-tp1 and fresh corresponding
source-gpu3-r1-stage/source-gpu3-r1-submit intents. Declaring a slot never starts
a model automatically; verify its actual profile and forward progress later.
For each declared node10 SLOT, e.g.:
python3 -m $M.queued_generation_control slot stage --generation owner-v2 --enabled-slots "$NODE10_SLOTS" --slot tp4-0-4-6-7 --instance r1 --intent-id node10-tp40467-r1-stage --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.queued_generation_control slot submit --generation owner-v2 --enabled-slots "$NODE10_SLOTS" --slot tp4-0-4-6-7 --instance r1 --downstream-owner-ready "$DOWN_READY" --downstream-owner-ready-sha256 "$DOWN_SHA" --intent-id node10-tp40467-r1-submit --published-commit "$PUB" --published-ref "$REF"


The only additional model beyond the measured current5 topology is GPU7 TP1:
python3 -m $M.queued_generation_control slot stage --generation owner-v2 --enabled-slots "$NODE10_SLOTS" --slot gpu7-tp1 --instance r1 --intent-id node10-gpu7-r1-stage --published-commit "$PUB" --published-ref "$REF"
python3 -m $M.queued_generation_control slot submit --generation owner-v2 --enabled-slots "$NODE10_SLOTS" --slot gpu7-tp1 --instance r1 --downstream-owner-ready "$DOWN_READY" --downstream-owner-ready-sha256 "$DOWN_SHA" --intent-id node10-gpu7-r1-submit --published-commit "$PUB" --published-ref "$REF"
The other four TP1 slot names are exactly gpu1-tp1,gpu2-tp1,gpu3-tp1,gpu5-tp1;
use distinct corresponding node10-gpuN-r1-stage/submit intent identities.

Record true BF16/model4096 initialization, exact retrieval identity and two
actual forward observations including complete sealed batches after starting.
No synthetic GPU baseline is introduced. Model8h instance completion does NOT
stop the24h index; later r2 is a fresh intent only after r1 is terminal/leasefree
and real owner remaining lifetime suffices. Current sole read-only follower is
not duplicated or edited by these tools.

Paths that must remain on NAS10
-------------------------------
$B/config.json; all1322 inputs; existing sampling outputs, phase configs,
unsealed partial states/features, caches/corpus/index and bank seals. Keep all
$B/pool/jobs/job-*/attempt-* allocations/children/progress/failure/complete/logs,
model-init/KV profiles, original-stage manifests/bootstrap/path wrappers and
native launch/evidence directories. No tensor/checkpoint/FAISS copies to Mac.

Especially root original job:
$B/pool/jobs/job-shared5-1490214/attempt-1792522434235917966-1490214/
allocation SHA6c5a59cb1a2199b909071b829f20af0cb6991f9e1a240f715b14cb4b94c38dcd.
Keep preparation/shared-vram-native-control-v6; source5-model-sublease-
retrieval-v1,gpu1-v3,gpu2-v3; source5-tp1-replacement-gpu3/4/6/7-v1;
source5-model-tp4-to-tp1-v1; relay-consumer10-control-v1 (old source-context,
upstream-ready and0600 secret NEVER returned as value); oldparallel/TP1 control
directories; legacy-relay10-independent-v1; relay-primary10-SLOT-v1/v2/v3;
node10-native-failures-20261021-v1; node10-primary-v1-pre-model-terminal-20261021.
Keep all original directories/inodes; terminal proof hashing does not move,
rewrite or sweep model/index/whole trajectory files. Do not unmount globally.

New proof roots are preparation/cohort-rollover-v1-h100-10 and-h100-5:
plan.json, exact staged code manifest, collector-intent/receipt, collector.log,
complete.json. Source5 private namespace must be pinned by the collector before
root exit, so these remain shared after root stops. Local work receives only
small summaries/hashes. No new owner is admitted from a plan or CPU fixture
alone: COMPLETE means actual terminal states+two actual releases.

Implementation handoff
-----------------------
NEW files:rollover_terminal.py,rollover_terminal_control.py,
guarded_rollover_control.py,guarded_rollover_slot_control.py,
queued_health.py,queued_slot_consumer.py,queued_relay_owner.py,
queued_generation_control.py,queued_relay_slots.py,rollover_source_slots.py,rollover_paths.py,
rollover_retrieval_owner.py,test_rollover_terminal.py.
Latest focused local verification:6 tests passed in0.077s with
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest reproduction.operations.qwen7b_resample.test_rollover_terminal -v
Historical six-slot verification preceded this extension; the current renamed case is:
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest reproduction.operations.qwen7b_resample.test_rollover_terminal.RolloverTests.test_six_node10_slots_match_owner_entry_model_entry_and_actual_bootstrap -v
PASS1 in0.016s, including real generated bootstrap to fakeexec and matching
owner/model registry+sourceSHA. Initial fixture path assertion was corrected
for macOS /var versus /private/var before the successful rerun; no GPU test.
Later local import/main audit fixed queued_relay_owner import-time exports
(new six-slot registry is now explicit without mutating the old owner on
import) and restores delegated control bundle/REMOTE globals after main.
Three targeted cases passed in0.084s: import/main+all new handoff CLI arguments
parsed in plan mode with transport/publication disabled; all five owner/slot
bundle source_files/files hashes including final bootstrap checked; exact
staged verify_source_proof accepts the new actual terminal status and rejects
the old status. The existing six-slot actual-bootstrap case passed again.
Minimal repeat command (local CPU only):
ROLLOVER_CASE=reproduction.operations.qwen7b_resample.test_rollover_terminal.RolloverTests
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest "$ROLLOVER_CASE.test_import_contract_and_handoff_commands_use_new_registry_without_global_leaks" "$ROLLOVER_CASE.test_manifest_hashes_and_staged_source_verifier_accept_actual_new_terminal_contract" "$ROLLOVER_CASE.test_six_node10_slots_match_owner_entry_model_entry_and_actual_bootstrap" -v
Old V5 identity/controlled-proof masquerade, wrong
slot instance/allocation and missing retriever/active original model reject.
Namespace-path fix verification:10 focused CPU tests passed in0.243s. New real
temporary-file regression executes capture/collect/verifier/fake-systemd owner
controls in a private alias-only view, then renames the same tree into a host
physical-only view and repeats proof/owner-control admission. Physical records,
original inode and SHA stay bound; a visible wrong alias rejects without SHA
fallback. No live process/unit, GPU or remote operation is part of the fixture.
Minimal path-regression command:
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest reproduction.operations.qwen7b_resample.test_rollover_terminal.RolloverTests.test_private_alias_then_host_physical_view_capture_collect_and_owner_controls_keep_sha -v
Actual source/terminal/deployment acceptance remains a later root operation.

Future node10 GPU7 capacity declaration (local code only):
The default registry now has6 slots (current5 topology+GPU7TP1); GPU7 TP1 starts64seq/8192tokens/4096KVcap/1GiB reserve.
This does not alter the current five-slot capture or old TERMINALS. GPU7 is not
claimed to be launched, fitted or productive. Control/owner/model entry share
the new registry; primary lock proof and per-slot lease remain mandatory.
Current focused CPU command executes both GPU7TP1 and measured-topology TP4 generated bootstraps to fake exec:
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest reproduction.operations.qwen7b_resample.test_rollover_terminal.RolloverTests.test_six_node10_slots_match_owner_entry_model_entry_and_actual_bootstrap -v

Root final source check:10 focusedCPUtests PASS0.301s after finite6capacity and
historical importprofile fixes. Newly generated TP1 bootstrap explicitly uses
SEARCH_G1_PHYSICAL_WORLD_SIZE=4 only for historical4/8-only science imports;
actual CUDA_VISIBLE_DEVICES=7/--tp1 and perbatchworld1 remain unchanged. Both
TP1 and TP4 generated environments really imported contract in CPU subprocesses.
No futuremodel has been launched by these changes.

Read-only supervision of the future generation
----------------------------------------------
follow_guarded_generation.py and guarded_generation_read.py derive the exact
17-unit registry from the staged manifests: one source owner, eight source
model slots, two CPU relays and six node10 model slots. The default `plan`
action is local only. Only after real submission, `once` or
`follow --hours 4 --interval 300` reads native status and small NAS10 metadata.
It launches/stops no remote work, takes no locks and writes no primary state.
Its immutable local observations contain metadata only. Keep current77291 as
the sole current-cohort follower until the actual generation changes.

Whole-plus-next acceptance requires an actual own batch seal and a different
batch with a live, matching POSIX lock owner and GPU/unit binding. Historical
failures retain their original records and are classified only by exact
proof-path/SHA bindings, including the actual predecessor chain. New failures
still require root attention. A finite model completion before the global
bank is complete requests renewal; transport uncertainty never implies a
native exit. Full metadata counts request the final independent data audit,
not goal completion. Four focused CPU fixture tests passed; this is not a
claim that the new generation is running or has produced data.
Cross-shard batch identity uses the pair (shard,batch), and a submitted unit
that has not yet loaded remains NOT_STARTED rather than an inferred exit.

All full data, inputs and original logs remain in our own shaoshitong NAS10
directory. Shared aliases refer to this same retained bank. No foreign user's
disk and no Mac tensor/trajectory storage are part of this procedure.

Actual collector launch correction (08:20–08:26 CST)
----------------------------------------------------
Both original collector starts were attempted once. Source5 collector is really
active and pins the old private namespace; do not restage/restart that unit.
Node10's auxiliary collector exited1 before Python collection: its multiline
bash -c payload was serialized into transient unit lines. No sampling unit was
stopped and no drain was issued at this failure. Original transient unit inode
and bytes plus a journal export snapshot are retained on NAS10 at
preparation/cohort-rollover-v1-h100-10/collector-original-start-failure-v1/RECORD.json
SHA5345af224ba557081c4473b21cc883c6eb96c3e939bf2a2e7cb673ca1a7d1570.

The new node10 repair control executes the immutable staged native_paths.sh
by physical filename and explicitly passes Q7B_NATIVE_UNIT_INSTANCE. Use
--collector-instance repair1 on both stage and collect, with the SAME generation
rollover-v1 and original PLAN10 SHA. It has a distinct unit, intent, receipt
and log. Under the shared submit lock, every prior collector must have an
actual loaded/MainPID0/inactive-or-failed state and a known receipt. Original
consumed intents and failure files are retained; an active or unknown prior
collector rejects. Do not replay the original consumed collect action.

The same multiline-argv issue was found before starting three future owners.
NEW guarded/queued remote_source owner branches now carry a single-line Python
decoder plus base64 payload, preserving shell arg0/node/bootstrap and existing
Q7B_NATIVE_UNIT_INSTANCE. It decodes only after unit entry; source5 does not
need a host-visible wrapper path before the NAS10 private mount. All14 model
slot paths remain nsenter -> Python bootstrap filename. Six focused CPU tests
passed including real temporary multiline decoding and alias/physical proof
admission. No new GPU test was added and no live sampler source was changed.

Those NEW control files change all17 future bundle identities: preserve old
stages and stage all17 new exact manifests after publication, before any owner
or model submit. This paragraph supersedes earlier 'all17 stages ready' for
the corrected sources. The early collector window is now already in progress,
not an unexecuted08:55 schedule. After actual repair collector is live, request
node10 whole-batch drain, obtain trueDONE10, then retire exact source dependents
and root LAST; preserve the same bank and sealed prefixes throughout.
