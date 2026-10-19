"""Read-only resource wait for the qualified171 continuation; never dispatch."""
import json
from pathlib import Path
import time
from .control import OUT, remote

# Actual owned-process NVML peaks over the completed actor171 phase (56 samples).
# This is an observed footprint, not an invented idle-node or45GB policy.
OBSERVED_MIB = [38546, 36690, 37428, 41628, 52502, 32836, 39296, 44202]
SCRIPT = """import json,time,subprocess,socket
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(dict(observed_unix=time.time(),node=socket.gethostname(),gpu=raw)))
"""


def save(name, value):
    root = OUT.parents[1]
    destinations = (OUT, root / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',
        Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018'))
    for path in destinations:
        (path / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    while True:
        observations = []
        for node in ('h100-5', 'h100-10'):
            evidence = OUT / ('recovery171-capacity-%s-%d-transport.json' % (node, time.time_ns()))
            try:
                value = json.loads(remote(SCRIPT, evidence, target=node, established_controller=True))
                cards = []
                for line in value['gpu'].splitlines():
                    fields = [item.strip() for item in line.split(',')]
                    index, used, total = int(fields[0]), int(fields[2]), int(fields[3])
                    cards.append(dict(index=index, uuid=fields[1], used_MiB=used,
                        total_MiB=total, free_MiB=total-used,
                        observed_actor171_MiB=OBSERVED_MIB[index]))
                cards.sort(key=lambda card: card['index'])
                if [card['index'] for card in cards] != list(range(8)):
                    raise ValueError('Require exactly eight physical H100 indices')
                value.update(cards=cards, evidence=str(evidence),
                    observed_footprint_fits=all(card['free_MiB'] >= card['observed_actor171_MiB'] for card in cards))
                observations.append(value)
            except (RuntimeError, ValueError, OSError) as error:
                observations.append(dict(node=node, observed_unix=time.time(),
                    status='CURRENT_RESOURCE_UNKNOWN', error=str(error), evidence=str(evidence)))
        result = dict(observed_unix=time.time(), source_complete_outer=171,
            observations=observations, excluded_nodes=['h100-11'],
            h1008='returned and not responding; no submission to this node',
            training_submitted=False, native_action='none; read-only capacity observation',
            capacity_scope='actual actor171 footprint, not exclusive-node admission; recheck before dispatch')
        save('RECOVERY171_CAPACITY_LATEST.json', result)
        print(json.dumps(result), flush=True)
        if any(value.get('observed_footprint_fits') for value in observations):
            save('RECOVERY171_CAPACITY_CANDIDATE.json', result)
            return
        time.sleep(1800 if all(value.get('status') == 'CURRENT_RESOURCE_UNKNOWN' for value in observations) else 120)


if __name__ == '__main__':
    main()
