# Fourth matched factorial cell L21/P80 — ready, not submitted

Remote entry: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1/layer21-prefix80/training.sbatch`

Entry SHA: `71bb3bc2cc92647282dcc7ae8fba669c1ec1c8412876c8d6505ea86138ac8f8c`; 1337 bytes. Binding: `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1/layer21-prefix80/factorial-ready.json`, SHA `81e9dad0e26d55891822c346f683a781da4786669990a2478850be4efea0f5d7`, 5612 bytes.

The existing immutable runtime commit5a3cfa1588c883d1c816d77ff95191acaa71eb35 remains unchanged. Only the new layer21-prefix80 directory receives an entry/binding. The entry copies L27/P80 and changes only the layer, job name, output/lock path, initial partition to normal/acd_u and walltime to2h. It retains twoGPU ranks,12CPU,128GiB, global128, local64, micro8, accumulation8, three epochs/192 updates, LR5e-6 andEMA.99. Head seed9137, same frozen official actor, fresh block21 andsameP80 head initialization. Captured features are postblock20, computed online from the same true causal Prefix80 journal. The exact existing8192 G10 fixture, three epoch G5 plans and fixed2048 identities are bound by SHA; no new framework or feature extraction was introduced.

Stage controller and root-owned one-time submission control were published and remote SHA verified before staging: commitd37a10a157253bdf1cdb50ba4b96121298f2c1cf on codex/q7b-online-L21P80-factorial-20261022. Python compilation and remote bash syntax passed. No model/feature checkpoint load, GPU process, Slurm submission or mutation occurred in this staging operation. Existing main, uniform control and P80 pilot remain untouched.

Root alone can submit by passing the following script to the existing balanced-control.py:

`reproduction/reports/qwen7b-algorithm-improvement-20261021/online-layer-prefix/submit-l21p80-factorial-once.py`

The prepared control validates metadata, checks same-name jobs and its own lock, records intent before submit, checks the held job identity, then releases it. Existing receipts are returned without resubmission. An unresolved intent requires same-name/sacct reconciliation, never a blind retry. The stage controller never calls the submission control.
