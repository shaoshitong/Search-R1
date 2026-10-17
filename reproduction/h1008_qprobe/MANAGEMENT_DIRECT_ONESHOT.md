# Task-local direct observation mode

The current Search-G1 readers accept a connection receipt containing:

```json
{"control_path": null, "compact_handshake": true, "transport_mode": "direct_one_shot"}
```

This opts one invocation into the existing `mgmt-host` alias, identity, physical
route and strict known-host checks, using curve25519, the verified ed25519 host
key type, aes128-gcm, hmac-sha2-256 and no compression. With no explicit task-owned
socket it disables SSH multiplexing for that invocation. It does not edit global
SSH/network configuration, add an asset relay, or reuse another task's socket.
The inner authenticated H100 route and four-node whitelist are unchanged.

An absent/false flag retains the previous transport argv. There is no automatic
fallback or retry; connection failure means unknown observation, not terminal
training, and never authorizes restart, duplicate submission or runtime edits.
Ordinary, legacy compatibility and explicit task-owned socket modes remain.

This mode was motivated by authenticated masters that subsequently failed to
open a new channel, while a separate single connection read original3433 joint24
and ongoing25. Those readbacks do not prove continuous connectivity or isolate
the network cause. Keep native run/outer/rank evidence and failure timestamps.
Only small control evidence is returned; models, data and raw trajectories stay
on the H100 shared storage.
