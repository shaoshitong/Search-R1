# Ablation task SSH candidate, 2026-10-15

The unchanged management endpoint is intermittently reachable. Previously accepted task masters actually exited 255; failed observations do not establish a training exit. The original C1 study continues independently in allocation 3393.

`release/recursive_bran_a/h100_migration/supervise_task_transport_noqos_v1.py` copies the already published local supervisor at `12ae5670c80a311f646837be7bddee5a697bf0c6`. The only operational change is the explicit OpenSSH option `IPQoS=none`. Its description also identifies it as a candidate. All alias, interface, authentication, strict host-key, keepalive, timeout, bounded retry and child-wait behavior remains identical. The original source is preserved.

This option is a connection hypothesis, not an established root-cause fix. Another authorized H100 task recorded a successful connection with this option plus a Kex option; that observation does not isolate either option. This candidate changes IPQoS alone. Actual hostname and Node reads are required before accepting a socket, and subsequent actual child exits and elapsed lifetime must be preserved. No permanent stability or causal improvement is claimed in advance.

Only a task-owned local reconnect supervisor may be replaced, after recording its actual child wait and exact local ownership. No GPU process, remote training state, scientific source, model, data, evaluation schedule, global SSH/VPN/auth configuration or another task's private socket is changed. This code creates only private task sockets and diagnostic logs.

Publish and read back these exact source bytes before using the candidate. Continue the full ablation goal and the original training process; reconnect success is not training completion.

## Recorded trial

The candidate produced a successful management hostname request and actual Node reads, but its first accepted SSH master subsequently exited with actual wait code 255. The recorded interval from the first successful hostname observation to the child wait was 460.362 seconds. This interval is not a measurement of the exact network-disconnect instant. Private evidence SHA: `d246e07f9ed7ae732316fd811cf10b68033316692e765fff366ce2b5ebb17b54`.

Thus `IPQoS=none` alone did not eliminate the observed disconnect. One trial does not establish a comparative lifetime improvement or the underlying network cause. The published candidate continues reconnecting to the unchanged endpoint; it must not be described as a proven fix. Fresh reads after reconnect confirmed the same original C1 owner/source and advancement from native outer64 to65. No GPU training restart occurred.

The deployed local helper remains source commit `7304a39d3e13bff807693068dd99327b93b32e33`, file SHA `510c63da956c9c4f993349a7b02c5a24fc6a620a0ecc1cab70cd43b75241f319`. This documentation update does not change its code or the scientific training source.
