# Independent read-only MAE observations

The original full169028 MAE arm was submitted once by root as job749569 from source7ac274ac8ff5dea0a12dfd44cfb03fe57491458f. It requested two GPUs, ten epochs, global128/local64/micro8/accum8, LR5e-6 and EMA0.9995. These observer scripts do not submit or change jobs, loss, runtime, partitions, QoS, other arms or data.

The first actual read found it PENDING. A later read established **FAILED 1:0 after9seconds on ACD1-13**. Both ranks raised `RuntimeError: CUDA unknown error` inside initial `torch.cuda.set_device`, before process-group initialization, writer locks, controller creation or any optimizer update. No checkpoint or evaluation was produced. This is a CUDA initialization failure; it does not establish a loss/data/model defect. Root was notified for device/allocation diagnosis and any authorized recovery submission.

`actual-mae-initial.json` and `actual-mae-latest.json` preserve actual accounting, the exact source/job identity, rank failure tracebacks and bounded log tail. `observations-mae.jsonl` is this observer's separate small journal. No final4096 data was read.

The bounded follow-up script stops on actual failure. If root authorizes and uniquely submits a corrected replacement, its new exact job/source identity must be recorded separately before following it; the original failed job must not be represented as still running or silently reassigned.
