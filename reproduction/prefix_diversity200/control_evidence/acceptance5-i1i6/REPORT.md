# Fresh I1-I6 checkpoint5 CPU acceptance

PASS, process854187 exit0. Actual immutable joint5 SHA `1f4ba1d50a15f5b22ccc705b3237c184471052f5b61408e1999d621ab55d4f50`. Both actor/probe writers had stopped successfully before once intent/deployment. Runtime `68d2d131aee91d194c487f54a0204f53397d87ea`, executed reader `09608ca28a6c18326bf9c01bf1c31fa26323e49ad6ba49f973531330aeb646da`, published execution commit `ecabb7f31848c13f0b6ecdae6d03a452af549f61`.

Read all HF435 FP32 tensors and every actual HF file SHA/tokenizer, all8 native actor optimizer states (100 Adam/rank, scheduler5, next6), saved rank RNG, probe100/14 FP32 model+Adam states/8 private RNG, current strict driver/pool boundary (cursor25600, next6), pinned source/splits, original G5 native prefixes/ownEM, stable Top1024 and population-EM-variance Top512. Original logical-pair weights were recomputed from native masks for all8 rank20-write ledgers; probe1024/5120 natural one-pass20 synchronized writes and true-mean denominators passed. Prior warmup CPU proof was reused; no old training checkpoint weight file was opened.

Selected5 learned tensors compared to official BF16 values promoted toFP32:6110/6656 elements differ, maxabs3.0994415283203125e-6; no fallback required. Official full weight SHA was not repeated; selected official small tensors only, during the same learned HF traversal.

Three live progress observations plus terminal: log225→5469→6669→9619 bytes, CPU25.35→55.14→70.52 seconds before terminal. Full once process wall94.9861s, user69.697787s+system25.256703s, maxRSS4543524KiB. Runtime limits Memory12G/CPU100%/Nice19/IOWeight10/thread1/empty CUDA were recorded in deployment. Training was not modified.

Remote proof: `/mnt/data/shaoshitong/search-g1-prefix-diversity200-20261017-4a1cc2d500fe/preparation/joint-milestone-0005-independent-full-sha-cpu-once-v1.json`; SHA `4cdaa4c502718338c0262dae7f0080fb79dd876e8a944175167d96b4c1af5fb1`. Local CPU_FULL_PROOF.json was reserialized and independently matched that exact remote proof SHA. Model/tensor/native data remain remote; local evidence contains only small JSON receipts.

CPU restore does not prove CUDA/executor exact resume or an independent backup. Actual original QA5 had not completed at CPU read; shared BF16 source contract was verified and no new QA was run. Parameter-change sampling does not compare all model parameters.
