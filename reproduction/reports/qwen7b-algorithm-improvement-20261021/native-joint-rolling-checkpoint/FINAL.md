# Native joint rolling checkpoint helper

API: `save_native(base,state,pin=False)` in `native_joint_rolling_checkpoint.py`. The main runner supplies the existing six-state native blob plus its accepted sampler state/hash and explicit data cursor. The helper preserves14418+cursor Adam clocks and the original four-rank RNG archive.

The inherited saver accepts a staged full file through CPU raw/EMA/Adam/buffer/RNG reload and SHA. The helper independently reloads and checks sampler/cursor metadata, then publishes a stable accepted file, atomically replaces the `rolling-latest.pt` hard-link alias and `latest.json`, and removes only the previous rolling body after publication. Pins remain immutable. A publish interruption retains the old valid latest body and the new verified body with a transaction record; unresolved transactions are rejected rather than silently overwritten.

Four synthetic CPU tests passed remotely: **4/4**,0.963s test execution,12.82s including cold imports,observed RSS521498624B,CPU affinity0/1,CUDA uninitialized. They cover full next-update/sampler restore, pin/rolling lifecycle, bad-state refusal, and interruption during final metadata publication. A separate stdlib readback verified the result/log file SHA.

This is synthetic transaction qualification, not a real GB-checkpoint performance test. No real model/checkpoint was read. After745177 ended, normal/acd_u rejected GPU0 CPU submissions and debug had no available GPU0 configuration; no new allocation was created. Root then explicitly authorized bounded login synthetic tests. The test harness uses an explicitly named nonallocation sentinel solely to mock the helper's Slurm guard. Initial cold-import deadlines/partial logs were preserved; moving lazy optimizer-module imports into the authorized cold-start phase yielded the complete result. Real six-state CP1 qualification remains the independent745177 acceptance.

The helper's scientific implementation did not change during harness qualification: SHA`c4e7a06a6c5d49ddc37d493b0a887423d22092ddebed13e87b4b11b457fae29a`. No training source or GPU was changed by this task.
