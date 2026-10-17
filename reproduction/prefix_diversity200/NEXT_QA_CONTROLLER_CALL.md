# Next milestone / final QA controller call

Only after a real complete joint100/150/200 or actor200 final51713 boundary exists: read its actual checkpoint, source, config and joint SHA identities. Do not reuse checkpoint50 identities or rerun accepted CP50 CPU acceptance.

If native QA fails, keep training running and select an existing authorized same-cluster node with capacity for approximately 23 GiB per rank. Retain model/data/index assets on shared NFS. The pinned controller route allows exactly h100-5 → 10.33.0.5, h100-8 → 10.33.0.8, h100-10 → 10.33.0.18, h100-11 → 10.33.4.14; gateway is 10.33.0.18.

Use the existing published QA deployment/submission code, with a fresh namespace and unique evidence files:

```python
import json
from pathlib import Path
from reproduction.h1008_qprobe.management_transport import remote
from reproduction.prefix_diversity200.qa_standalone_control import deployment_script, submit_script
from reproduction.prefix_diversity200.qa_standalone_observe import observe

# actual_config and qa_commit must bind the real completed boundary.
node = actual_config['node']
deployment_receipt = json.loads(remote(
    deployment_script(actual_config, qa_commit), deploy_evidence,
    target=node, established_controller=True))
Path(deployment_receipt_path).write_text(json.dumps(deployment_receipt))
# Reconcile deployment receipt and any existing intent before the one submission.
remote(submit_script(deployment_receipt['base']), submit_evidence,
       target=node, established_controller=True)
observe(root, deployment_receipt_path, connection=canonical_connection_path)
```

The observer chooses `connection.transport_mode`, defaults to the task's `ESTABLISHED_CONTROLLER_CURRENT.json`, and preserves an explicit legacy connection option. Established mode uses the pinned existing `control.py.route(target, '/usr/bin/python3 -')` with no socket, SSH overrides or route fallback. Follow real QA progress through terminal exit, independent saved-output acceptance and exact unit cleanup; no evaluation is started by this note.
