# Native process-exit evidence, separate from training progress

The existing detached leaders have PPID1. A completed pipeline marker and a disappeared PID do not independently establish an OS exit status of zero. The training entry points remain frozen and running.

The host Linux proc connector was first qualified using two tiny owned CPU children. Their actual parent wait() return codes were 0 and 7, matching the kernel raw wait statuses 0 and 1792. Kernel parent notification SIGCHLD=17 was not misinterpreted as the termination signal. No model, GPU, training process signal, ptrace or global kernel setting was used.

One passive exit-only collector then subscribed to kernel events and bound both existing leaders by PID, birth ticks, SID, UID, submitted run/config, boot ID and current command. Actual submit receipts are read from the current run namespaces; the intentionally replaced old B leader is excluded. Bound pidfds independently confirm exit of the same process identity. Events for other processes are discarded without being stored.

At the first actual status read, the collector is armed and the original birth is live. No trainer has exited yet, and no actual exit-zero evidence is claimed. Later per-arm proof requires the real kernel event and readable bound pidfd. A lost/unavailable event records unknown status, never an inferred success. This collector cannot replace QA, checkpoints, retention or owned resource cleanup.

The one existing progress observer remains unchanged. This collector does not poll iteration metrics or write the primary state.

| Arm | Bound leader | Birth ticks | Actual exit status |
| --- | ---: | ---: | --- |
| A | 128824 | 704780598 | Pending |
| B | 757735 | 709771046 | Pending |

Remote collector root: `/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-native-exit-evidence-20261018/h100-11-24bbc2ee-c187-492f-bce3-a3afaa2b08e6/ab-leaders-128824-757735-once-v1`.

Small first actual status SHA256: `acc05c04a9b9e731821c00817e0514107cd5767b4c53b253b37184473920b734`.
