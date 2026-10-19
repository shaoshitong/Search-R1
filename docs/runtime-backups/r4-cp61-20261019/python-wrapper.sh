#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4/repo:/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4/env/overlay:/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4/retriever-gpu-lib:/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/env/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/retriever-python310-candidate/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONHOME
exec /data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/retriever-python310-candidate/bin/python3.10 "$@"
