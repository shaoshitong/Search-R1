# Task SSH lifetime diagnostics

`release/recursive_bran_a/h100_migration/supervise_task_transport.py` runs only
on the local control machine. It uses the existing `mgmt-host` alias and host
keys, with a private short socket, a small private SSH log and the actual
foreground SSH child wait status. It changes no global SSH/VPN/auth settings
and sends no remote training commands or signals. Production science and its
deployed source identities remain unchanged.

```sh
python3 -B release/recursive_bran_a/h100_migration/supervise_task_transport.py \
  --state /absolute/private/control/foreground-ssh.json \
  --logs /absolute/private/control/foreground-ssh-logs --attempts 20
```

The initial state is a candidate, not accepted connectivity. Independently
require `ssh -S SOCKET -O check mgmt-host`, an actual manager hostname read and
then fresh node observations before using the socket as accepted transport.
Use the exact active state socket; each failed attempt has a different one.
An SSH wait code, missing socket or observation timeout never proves a training
exit. Do not restart training from a transport failure. Preserve the actual
exit journal and SSH diagnostic log; they are private local audit files and
must not be committed. The longer task keepalive is an unproven resilience
candidate until an observed outage demonstrates that it helps.
