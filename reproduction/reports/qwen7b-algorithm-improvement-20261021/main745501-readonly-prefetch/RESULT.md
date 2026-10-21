# Finite read-only input prefetch: actual result

Main job745501 retained its original PID, configuration, source and all of its SHA checks. One CPU2 /2GiB /0GPU step used8 read threads and a1GiB maximum ahead window. No tensor was decoded and no model was loaded.

ABBA disjoint-file range measurements: sequential228/247MB/s; parallel674/805MB/s. The real sidecar ran226.43s, full-SHA checked8590 files and read89,154,339,899B. Peak RSS158,956KiB. Main rchar117.079 to205.842GB: observed 434.5MB/s; between the first two observations 424.1MB/s. These are actual runtime observations, not a cold-cache controlled causal training-speed comparison.

Final pointer3951/3952; automatic exit after main stopped reading feature payloads. Sidecar step745501.7 COMPLETED0:0(3:47); main745501 remained active. No extra GPU job, optimizer update, held-out payload or final evaluation access.
