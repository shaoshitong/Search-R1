# Finite source inventory after the accepted366 batches

Fresh NAS10 metadata observation at **2026-10-21 00:46:33 UTC** found128 additional sealed canonical G5 batches beyond the already accepted244+122 batches. They contain **16384 questions and81920 trajectories**, comprising **16256 HotpotQA and128 NQ questions**. These remaining source batches have not been exported, downloaded or accepted on HPC3 by this task.

| Source status | Batches | Questions | G5 trajectories | Domain questions |
|---|---:|---:|---:|---|
| Currently sealed overall | 494 | 63232 | 316160 | Overall domain recount not required for this delta |
| Already accepted migrations | 366 | 46848 | 234240 | Frozen prior244+new122 |
| New sealed metadata-bound difference | 128 | 16384 | 81920 | Hotpot16256, NQ128 |
| New with existing independent audit still matching all file fstats | 100 | 12800 | 64000 | Hotpot12672, NQ128 |
| New without a current matching independent batch audit | 28 | 3584 | 17920 | Hotpot3584 |

The canonical1322-input/169028-question source is **not complete**:828 batches remain unsealed and the root completion marker is absent. The independently audited metadata summary reports466 verified batches/59648 questions/zero failed batches, but was last written at00:27:41 UTC. Its own pending count1 is stale relative to the current sealed inventory and must not replace the fresh28-batch difference above.

The source retains a historical root `failure.json` with `RuntimeError`, timestamp **2026-10-20 16:14:35 UTC**, SHA`314647bc03dc3f48c1a70e7177ba37f178e166f8a28150800bd3c8faffb90a5f`. It predates the new seals below. This observation does not inspect the writer process or infer a current exit from that historical marker.

Oldest new seal:2026-10-20 22:54:25 UTC. Latest new seal:2026-10-21 00:44:41 UTC. Existing independent proofs for the new audited subset span23:21:06 UTC to00:27:38 UTC. All times refer to actual metadata fields/file mtimes, not an estimated training or sampling rate.

The exact accepted366 baseline was rejoined to the immutable source244 export and source122 complete/plan, with original input/seal SHA, question count and trajectory count checked for every previously accepted122 batch. New batch paths are the strict set difference; duplicate or changed accepted batches fail closed. Original run, model revision, seed20261021/source1729, Prefix20/layer26 and actor-update0 contracts match the established source.

Remaining batch metadata digest: `d2453003c9061af5309e674c6dcefe38ba19dae24b27da734093275b6e7e8edb`.

Full inherited sealed-inventory plan digest: `4cd758d233eb5f5f47203be7b308745dd4d8d3a872729fbc004e75557161ae26`.

Canonical source config SHA: `856a5580c9faf6968aee9535bbc600d9c07f0b018dc13e85611fa76105bba789`; input manifest SHA: `445fc6914ab291bbdad52e1200642653b21c2dd62d3eaa7741631bcf61f71bfd`; independent-audit summary SHA at both actual reads: `c073ad51f0fdb07176af570f4f6c90c6a9ed084dbadfad0073f17fa3e2467ab4`.

The published source-only worker ran1.405 seconds on two CPUs within the3GiB bound; total control time7.989 seconds. The focused metadata difference fixture checks missing/changed accepted seals and exact subtraction. Source code was pushed and remote verified before invocation (`42e3f552`). All batch records remained in remote RAM; only aggregate metadata returned to Mac. No prefix/full/own-EM payload, model, checkpoint or feature tensor was opened. Existing batch audit proofs are inherited only while fstats match; this is not fresh payload SHA or tokenizer strict-EM revalidation. No export, upload, download, GPU, peer-process operation or peer-thread message occurred.
