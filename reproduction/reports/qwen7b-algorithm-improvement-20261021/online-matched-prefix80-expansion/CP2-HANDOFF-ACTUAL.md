# P80 CP2 handoff actual, 2026-10-23 CST

The first attempted STOP --full targeted the batch shell rather than freezing GPU step .0. The CP1 handoff guard caught later updates and prevented cancellation or rollback. CP1 transaction and hold receipts remain explicitly historical.

Both jobs then saved all 128 real Adam updates in CP2. Actual step .0 STOP was requested. L15 rank processes and job cgroup freezer were observed FROZEN at cursor128; Linux task state D in a frozen cgroup is not evidence of a storage stall. L18 reached CP2 before TIMEOUT; both writer locks were independently free and no later progress/evaluation existed. No partial epoch optimizer updates were discarded.

CP2 CPU acceptance passed independent archive SHA, separate raw/EMA strict loading, 46 Adam parameter states at step128, finite first/second moments and serialized two-rank RNG. GPU RNG restoration/next update remained untested at that acceptance stage.

L18 reused already-held successor749754; L15 successor749782. Both are independent two-GPU90min jobs, no dependency, exclude known CUDA-init failure nodeACD1-13. Both resume cursor128 from exact CP2, finish missing E2 evaluation before E3 updates. Old source/weights retained.

Fresh direct SSH returned Connection closed; the pre-existing approved campus route via hkust3 succeeded. No Mac proxy/network configuration/model transport used.

Source/controller publication71528d62aa42fac81b492a0616edee2054740e4b was pushed and ls-remote verified before successor release. Runtime74178e17320d2a3d44eca00766c3061330958bee was already independently pushed, staged and SHA checked.
