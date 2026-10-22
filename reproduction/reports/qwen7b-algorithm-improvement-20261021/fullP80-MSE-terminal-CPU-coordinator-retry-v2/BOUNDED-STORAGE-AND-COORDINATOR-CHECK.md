# Bounded shared-storage and coordinator check

Actual observation: 2026-10-23T06:18:02.438673+08:00 (2026-10-22T22:18:02.438673+00:00 UTC).

- Shared filesystem `/data` (`/LK02`, dpc) has **11,899,327,283,187,712 B available (11.90 PB decimal)**, 41% used. This is filesystem free space, **not mou176 account quota**.
- `/bin/quota -s -u mou176` returned code 1 with empty stdout/stderr within its 10-second bound. Account quota remains **unknown**. No extra quota search was performed.
- Job 750189: 3 published epoch CPs, **11,332,205,244 B**; 7 remaining saves estimated **26,441,812,236 B**.
- Job 750190: 5 published epoch CPs, **18,887,008,420 B**; 5 remaining saves estimated **18,887,008,420 B**.
- Combined remaining CP saves: **45,328,820,656 B (45.33 GB / 42.22 GiB)**. No temporary CP files were present. Estimates cover CP files only; other job outputs and account quota are not inferred.
- Actual workers remain healthy: v1 PID **3336295/start1408487**, poll137, job750189 both ranks E4/4736; v2 PID **29786/start1771045**, poll17, job750190 both ranks E6/7025. Each actual identity and own FLOCK was confirmed. No new failure, CP tensor FD, common CPU mutex FD or CPU child receipt.

Only bounded metadata reads and direct-directory CP stat were used. No recursive tree scan, CP hash/load, deletion, training/worker/Slurm mutation or final4096 access.
