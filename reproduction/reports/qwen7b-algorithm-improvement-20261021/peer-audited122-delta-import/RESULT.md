# Audited122 delta: transfer and strict-EM ingestion accepted

The exact published subset of122batches was imported independently: **15,616questions /78,080genuine new-seed trajectories**, NQ12,300 +HotpotQA3,316. It did not rediscover or export other newly sealed batches. The source169,028question generation remains outside this bounded completion claim.

- Source snapshot: `audited122-delta-20261021-673b8cb6`;8archives/722,391,040bytes/1,734members/719,345,030member bytes.
- HF pinned catalog: `30efd6dd3abaa6cc4cfe5587c2e57eae06ba3000`, catalog SHA`3a24be0718992cbb3d2f2f69e0c417a7c1f67198d69c343511d0147da34efe54`.
- H100→management bounded RAM→private HF→HKUST2→HKUST3→HPC3, with source/HF/per-hop archive checks and fresh shared-storage member SHA. Independent final readback covered all8archives +1,734members in41.01seconds and proved zero old244 overlap by batch/qid/question-hash/authentic trajectory IDs.
- Receiver complete: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-h100-20261021/audited122-delta-20261021-673b8cb6/complete.json`; SHA`6397e1c7137a4da66c45c3ba145603059dd67d37734019056072cb22726fabd5`.

The unchanged production ingester then recomputed own strict EM and verified complete causal contexts, question/token identity, and real old5+new5 seed namespaces. Its final accepted training bank has **15,616questions /156,160authentic trajectories**. Diagnostic512/selection2048/final4096 identity exclusions remained enforced: `{"diagnostic512": 0, "frozen_final4096": 0, "selection2048": 0}`; additional hash exclusions `0`. No synthetic repacking, duplicate-seed substitution, or validation/final scoring was performed.

- Groups: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed-ingest-audited122-delta-20261021-673b8cb6/raw-g10/groups.json`, 23196265bytes, SHA`c289201fd0b5c9c22289817bef94e56baa60b35b806c5aa7872b1624ea21f381`.
- Source events: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed-ingest-audited122-delta-20261021-673b8cb6/raw-g10/source-events.json`, SHA`d6d2de8c275c1a444071066edc993c90fdefae2afd2eeb8bf98819d6124358ac`.
- Ingest complete: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed-ingest-audited122-delta-20261021-673b8cb6/complete.json`, SHA`13f7e6a24699dd9858387210552a4d922ab521bd9a9c7c99648dab147f36048e`.
- Independent own-ID/ordered-EM/exclusion/histogram join: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed-ingest-audited122-delta-20261021-673b8cb6/independent-final-audit.json`, SHA`74490ebef994ed17723c87725cc515127922c34588d22d37aacb6cb1dcb878d1`.

Actual verification progressed 11,040→31,840→53,760→74,560→99,840 trajectories before the terminal result. The entire source subset/old244 identity boundary and all payload hash checks were independently accepted. Initial real fsync probes confirmed HKUST2 `/data` ext4, HKUST3 `/backup` ext4, and HPC3 `/data` `/LK02` dpc shared storage; probes were removed. HF used the previously owned HKUST2 loopback17898 exit after actual private archive Range verification. Mac carried only code/configuration/small aggregate receipts; credentials stayed on stdin/process RAM.

Original warmup had seen the canonical169,028training questions. This is newly imported genuine sampling data, **not an unseen-question claim**. No feature tensor, actor/probe update, GPU request, or modification to active training datasets was made. New raw data remains isolated for future explicitly selected use.

Immutable runtime `e7cedec11904271d748f75fe60df3c33b85e1d45`; operator revision `4cbe640c83e98632fc8c7357a3d659bb2f85f656`; independent join source `88d96c4b9b053682243a9dd27d724fba9abcf63c`. Code and all small receipts are backed up on `codex/peer-audited122-delta-transfer-20261021`.
