# Task SSH reconnect backoff

The management server returned `Exceeded MaxStartups` during the C2 observation outage. This is transport evidence, not a training exit.

`supervise_task_transport_backoff_v2.py` preserves the previous task-only SSH options and actual-child wait journal. It waits 60 seconds after ordinary connection failure and 120 seconds after an explicit MaxStartups banner. Both delays are configurable within bounded ranges. The state records the next permitted attempt time.

The helper does not run remote commands, touch training, change global SSH/VPN/authentication, or imply a successful hostname/node read. A live socket still requires a successful bounded remote read. The original supervisor source remains unchanged.

Before replacing a local retry process, verify its UID, PID, command and child ancestry; terminate only that task-owned local reconnect supervisor and its SSH master. Preserve its state, logs and actual exit records. Deploy the new helper only after normal Git publication and FETCH byte verification.
