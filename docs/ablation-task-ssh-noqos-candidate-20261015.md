# Ablation task SSH candidate, 2026-10-15

The unchanged management endpoint is intermittently reachable. Previously accepted task masters actually exited 255; failed observations do not establish a training exit. The original C1 study continues independently in allocation 3393.

`release/recursive_bran_a/h100_migration/supervise_task_transport_noqos_v1.py` copies the already published local supervisor at `12ae5670c80a311f646837be7bddee5a697bf0c6`. The only operational change is the explicit OpenSSH option `IPQoS=none`. Its description also identifies it as a candidate. All alias, interface, authentication, strict host-key, keepalive, timeout, bounded retry and child-wait behavior remains identical. The original source is preserved.

This option is a connection hypothesis, not an established root-cause fix. Another authorized H100 task recorded a successful connection with this option plus a Kex option; that observation does not isolate either option. This candidate changes IPQoS alone. Actual hostname and Node reads are required before accepting a socket, and subsequent actual child exits and elapsed lifetime must be preserved. No permanent stability or causal improvement is claimed in advance.

Only a task-owned local reconnect supervisor may be replaced, after recording its actual child wait and exact local ownership. No GPU process, remote training state, scientific source, model, data, evaluation schedule, global SSH/VPN/auth configuration or another task's private socket is changed. This code creates only private task sockets and diagnostic logs.

Publish and read back these exact source bytes before using the candidate. Continue the full ablation goal and the original training process; reconnect success is not training completion.
