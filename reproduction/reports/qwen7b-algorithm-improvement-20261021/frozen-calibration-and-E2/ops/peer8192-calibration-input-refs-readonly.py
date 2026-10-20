from pathlib import Path
import json,time
b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-frozen-calibration-scores-v1');c=json.loads((b/'config.json').read_text());print(json.dumps(dict(status='FROZEN_CALIBRATION_INPUT_SMALL_REFS_ONLY',groups=c['groups'],training_input=c['training_input'],identities=c['identities'],source_plan=c['source_plan'],model_names=[r['name'] for r in c['checkpoints']],time=time.time())))
