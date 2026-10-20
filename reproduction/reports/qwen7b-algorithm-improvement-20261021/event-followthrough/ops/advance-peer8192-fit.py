"""One finite reconciliation; run again only after a fresh observed state."""
import json,subprocess,sys
from pathlib import Path
C=Path(__file__).resolve().parents[2];repo=C/'work/qwen7b-probe-warmup-20261019';src=Path(__file__).parent/'reports/peer8192-feature-gate-current.json';s=src.read_text();v=json.loads(s[s.index('{'):]);root=v['capture_base'];args=[sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_peer_newseed_train','--control-root',str(C),'--action','advance','--submit-authorized','--raw-complete',v['raw_complete']['path'],'--identity-plan',v['identity_plan']['path'],'--validation-complete','/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/validation-fixedG5-hf-replay-20261021/capture/complete.json']
for i in range(4):args+=['--feature-complete',root+'/shard-%02d/capture/complete.json'%i]
r=subprocess.run(args,cwd=repo);sys.exit(r.returncode)
