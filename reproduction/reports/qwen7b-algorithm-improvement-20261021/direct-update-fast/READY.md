# Batched direct-Mixed updater

Published source `6731016ad5fa0b05a53d4390f8e4f5312a5140f3`; API `direct_update_fast(...same arguments as direct_update..., ema_decay=.99, *, micro_questions=8)` returns `(metrics,batch)`. Microquestions1/4/8/16 preserve B32 and the actual sampler's four authentic G5 views/question, quarter weights, two Adam clocks and EMA. The native block and existing two-layer group head are reused without modification.

Each chunk reads10*M own trajectories, performs one native encoding, gathers4*M chosen own G5 views, and backpropagates `per.sum()/(4*32)`. Per-view losses are restored to the original shuffled view order before the unchanged sampler observation. All updater finite/norm/confusion/loss metrics transfer in one compact CPU vector; the model's existing guards remain. Full step timing requires caller CUDA synchronization, since the function retains ordinary asynchronous optimizer/EMA semantics.

Two necessary synthetic CPU tests actually passed (2CPU/0GPU,14.565s including imports,5.628s tests, peakRSS506687488B). All four micro settings match scalar objective/gradient/Adam/EMA within declared floating tolerance; identical-micro complete-state restore reproduces the next update exactly. Batched BF16 GEMMs are not claimed bit-exact against a different microbatch. Fixtures were injected in a separate CPU interpreter's RAM, with no runtime deployment or real model/features read. There was no GPU submission or change to main746083.

The root/microbenchmark owners perform actual H100 qualification and choose any future immutable run integration. This module does not replace a live training function or change checkpoint code.
