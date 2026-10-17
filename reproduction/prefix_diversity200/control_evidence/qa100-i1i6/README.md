# Actor100 dev512 QA and grounding100 acceptance

Actual run `4a1cc2d500fe40979ef8ca4792959438` completed joint100, actor/probe2000Adam per rank. Original3433/frozen33ef training continued through complete101/2020Adam per rank; no training restart or runtime edits.

Native QA100 worker5 failed CUDA CUBLAS allocation while profiling; its original evidence remains remote. One isolated h100-11 unit reused accepted QA source `2b0a9f7b809d63880dcb0e30bebc96ab11267f45`:8TP1 BF16 workers, unchanged fixed512 strict scorer/interaction, full8GPU FP16 FAISS/E5 over local loopback. Existing40GiB initial/12GiB runtime headroom and36GiB own-memory guards remained. All8 workers completed64questions and exited0.

Saved-output independent CPU recompute:512 unique questions(NQ256/Hotpot256),210correct,EM41.015625%,F148.7718564%. CPU-only tokenizer reload and source/config/model/contract identities passed without new generation or HF weight reads. Aggregated F1 differed by1.11e-16,within existing1e-12 tolerance. Original scores were preserved. Native unit inactive/dead/PID0,cgroup absent/empty; no unreaped children.

Isolated QA wall144.178497s,physical8 wall opportunity0.320396655GPUh,CPU502.582737s(self+reaped children),Slurm GPU reservation0. Opportunity is not billed/exclusive GPU consumption and is separate from native training reservation. Independent CPU recompute9.562957s,user7.665587s/system0.930504s. CP100 native-state full-SHA CPU acceptance is a separate published packet `checkpoint100-cpu-v1`,not rerun here.

Grounding100 actually fit both heads using canonical300G5/1500raw trajectories(1497passed existing searched-valid commit-context filter). Eight healthy replicas adopted its manifest; completeactor101 actual reward receipts from all8ranks used that exact version. It is separate from the online prefix correctness probe.

Models,data,index,rawrollout and fullQA shards remain on remote/shared retained18 storage. This packet contains only small evidence. Full200 training,QA150/200 and final seven-dataset51713,grounding150,final costs/acceptance/owned cleanup remain unfinished.
