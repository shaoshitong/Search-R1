# Selected seed1 checkpoint CPU acceptance

Actual coordinator completed; process identity no longer live; no coordinator failure. Existing immutable76637 CPU validator entries were used one at a time under the shared8GiB mutex. No Slurm, model or optimizer mutation.

| selected checkpoint | actual job | attempts / Adam | raw/EMA tensors | CPU peak GiB | immutable CP SHA256 |
|---|---:|---:|---:|---:|---|
| current64 E5 | 749889 | 640 / 640 | 45 / 45 | 4.150 | `67d750089433893fd5184c9764f5ef2af46832d55cca9dd9b9f50a9b258ecb67` |
| mse-M5 E7 | 749857 | 448 / 448 | 45 / 45 | 3.948 | `99adde2521227b00727f482843f0d4cfc224c8b5c940559fe71300cc1658e4fd` |
| window2018 E10 | 749890 | 1280 / 1278 | 45 / 45 | 4.154 | `2ec741d9f3fca1b023c0ad0dfbdfb3928c2e2e2f9d1613745557e80f6406c215` |

CPU strict raw/EMA loading, per-parameter Adam moments/clocks, two-rank CPU RNG restoration plus serialized CUDA RNG shape, all ten question/own-trajectory plans, source hashes, evaluation binding, immutable CP stat/hash and cache roundtrip were checked. CUDA forward and GPU next-update equivalence were not tested by this CPU pass.

The earlier55-second SSH timeout created no E5/E7 receipts or live76637 loader. A durable coordinator subsequently ran the same entries successfully; the timeout was not reported as checkpoint failure.

Full actual receipt bundle: `three-selected-actual-accepted.json`; SHA256 545d11a2b7a60c4802000b8b0634321b7a3ef976f1ee1ef8aec63d82a529f445.
