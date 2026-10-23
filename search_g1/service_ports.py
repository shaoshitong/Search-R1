"""Independent local endpoints for DynRPO, preserving existing run defaults."""
import json
import os
from pathlib import Path


def ports(run=None):
    if run is None:
        base = os.environ.get('SEARCH_G1_BASE')
        path = Path(base) / 'logs/run-manifest.json' if base else None
        run = json.loads(path.read_text()) if path and path.is_file() else {}
    return dict(retriever=18110, probe=18112) if run.get('dynrpo7b_selection') else dict(retriever=18010, probe=18012)


def retriever_url(run=None):
    return 'http://127.0.0.1:%d/retrieve' % ports(run)['retriever']


def probe_url(route='judge_probe', run=None):
    return 'http://127.0.0.1:%d/%s' % (ports(run)['probe'], route.lstrip('/'))
