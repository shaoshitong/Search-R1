# User-authorized retirement of stopped R2 CP100 payloads

2026-10-18 human instruction: “之前训练到100iter的权重也可以删掉，50iter需要保留即可”.

The stopped source job was 729624, native accounting CANCELLED and every step terminal. Current job733342 is a distinct R3 continuation from CP50; its config explicitly names source R2 CP50. No live source writer was stopped for this deletion.

The operator first enumerated CP100 joint's four rank states, driver state, four Hugging Face safetensor shards and both aliases of the probe snapshot. Before unlink, every path had to be canonical, source-owned, and equal to its captured inode/device/size/mtime. Every inode was compared against all33 genuinely CPU-qualified CP50 files and the full CP50 actor/probe references. No matching protected inode was admissible. The intent was fsynced before mutation. Only `.pt` and `.safetensors` payloads were unlinked; configs, tokenizer metadata, producer receipts, logs, labels, rollout histories and metrics were retained. Both probe aliases were removed to release their shared inode.

Actual result:11 paths,39,209,008,988 unique-inode bytes removed. All33 CP50 qualification seals still matched exact inode/size/mtime afterward. This is not a claim of a second fresh full-SHA scan or an exact filesystem free-space delta under concurrent cluster usage.

Remote evidence (small JSON; no weights or credentials in Git):

- `/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2/preparation/user-authorized-old-cp100-deletion-intent-20261018.json`
- `/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2/preparation/user-authorized-old-cp100-deletion-complete-20261018.json`
- `/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2/outers/outer-000100/checkpoint-user-retired.json`

The last marker explicitly records that old CP100 is no longer loadable. Its historical score/receipt is provenance, not a surviving checkpoint. New R3 CP100 must be independently generated and accepted; it is outside this deletion scope. CP50 remains retained and loadable.
