#!/usr/bin/env bash
set -euo pipefail
BASE="${SEARCH_G1_BASE:?common8 base must be explicit}"
export SEARCH_G1_PROFILE=common8 RAY_DEDUP_LOGS=0
export SEARCH_G1_BASE="$BASE"
export PATH="$BASE/env/bin:$PATH" PYTHONPATH="$BASE/repo"
unset PYTHONHOME RAY_ADDRESS
export C_INCLUDE_PATH="$BASE/bootstrap-tools/python-dev/usr/include/python3.10:$BASE/bootstrap-tools/python-dev/usr/include"
export TRITON_CACHE_DIR="$BASE/cache/triton" TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor"
export TMPDIR="$BASE/tmp" HF_HOME="$BASE/cache/hf" XDG_CACHE_HOME="$BASE/cache"
export RAY_TMPDIR="$(python -c 'import json,os; from pathlib import Path; base=Path(os.environ["SEARCH_G1_BASE"]); print(json.loads((base/"logs/run-manifest.json").read_text()).get("ray_tmp_dir",str(base/"ray")))')"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8
export VLLM_ATTENTION_BACKEND=XFORMERS PYTHONUNBUFFERED=1
export PROBE_URL="${PROBE_URL:-http://127.0.0.1:18012/judge_probe}"
export ECO_TRUST_VARIANT=balanced ECO_TRUST_SIGNAL=probe ECO_PROBE_GATE=1
export ECO_BAL_GROUND=0.6 ECO_BAL_NOSEARCH_BONUS=0.5 ECO_W_COST=0.02
export ECO_WRONG_GROUND_SCALE=0 ECO_PROBE_BASE=0 ECO_PROBE_ALPHA_COST=0 ECO_USE_CF=0
export ECO_NO_SELF=1 ECO_PROBE_MAX_WAIT=300
cd "$BASE/repo"
G1_POLICY_EXECUTION_ROLE="$(python -c 'import json,os; from reproduction.common8.run_pipeline import supported_execution_role; from pathlib import Path; base=Path(os.environ["SEARCH_G1_BASE"]); print(supported_execution_role(json.loads((base/"logs/run-manifest.json").read_text()),base))')"
case "$G1_POLICY_EXECUTION_ROLE" in
  production|policy_safety_calibration_discovery|policy_safety_calibration_confirmation) ;;
  *) printf '%s\n' 'Refusing unsupported policy execution role' >&2; exit 1 ;;
esac
python -m reproduction.common8.probe_admission --verify
case "$G1_POLICY_EXECUTION_ROLE" in
  policy_safety_calibration_discovery|policy_safety_calibration_confirmation)
    # Freeze after the final ECO environment and real probe admission. The child
    # uses the original301 horizon; its bound phase dispatch stops after one outer.
    python -m reproduction.common8.freeze_calibration_plan
    ;;
  production)
    G1_CONTRACT_KIND="$(python -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["SEARCH_G1_BASE"])/"logs/run-manifest.json").read_text()).get("contract_kind","legacy300"))')"
    if [[ "$G1_CONTRACT_KIND" == fresh200_warmup57 ]]; then
      # supported_execution_role rejects a missing/wrong policy_safety_mode first.
      # Freeze real fresh0 probes and prospective candidates before this claim.
      python -m reproduction.common8.formal_inline_safety --prepare --base "$BASE"
    else
      G1_POLICY_DISCOVERY_BASE="$(python -c 'import json,os,sys; from pathlib import Path; print(json.loads((Path(os.environ["SEARCH_G1_BASE"])/"logs/run-manifest.json").read_text())[sys.argv[1]])' policy_calibration_discovery_base)"
      G1_POLICY_CONFIRMATION_BASE="$(python -c 'import json,os,sys; from pathlib import Path; print(json.loads((Path(os.environ["SEARCH_G1_BASE"])/"logs/run-manifest.json").read_text())[sys.argv[1]])' policy_calibration_confirmation_base)"
      python -m reproduction.common8.production_safety_admission --base "$BASE" --discovery-base "$G1_POLICY_DISCOVERY_BASE" --confirmation-base "$G1_POLICY_CONFIRMATION_BASE"
    fi
    ;;
esac
python -m reproduction.common8.claim_training
# Fresh official actor/reference and the same resolved recipe used by model admission.
python -m reproduction.common8.configuration 2>&1 | tee -a "$SEARCH_G1_TRAIN_LOG"
