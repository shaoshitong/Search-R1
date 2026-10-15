"""Unique control package: explicitly join legacy source and evaluator code."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def activate(source_root=None):
    control=Path(__file__).resolve().parent.parent
    root=Path(source_root or os.environ['SEARCH_G1_LEGACY_SOURCE_ROOT']).resolve()
    if root==control:
        raise ValueError('LEGACY_SOURCE_MUST_BE_ISOLATED')
    spec=json.loads((control/'reproduction/legacy_evaluation/protocol.json').read_text())
    for name,expected in spec['code_sha256'].items():
        path=root/name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('LEGACY_SOURCE_MISMATCH:'+name)
    # A regular reproduction package does not merge automatically across sys.path.
    package=importlib.import_module('reproduction')
    package.__path__=[str(root/'reproduction'),str(control/'reproduction')]
    sys.path[:]=[str(root),str(control)]+[p for p in sys.path if p not in (str(root),str(control))]
    for name,module in tuple(sys.modules.items()):
        if name=='verl' or name.startswith(('verl.','search_r1.','reproduction.common8.')):
            path=getattr(module,'__file__',None)
            if path and not Path(path).resolve().is_relative_to(root):
                raise ValueError('NEW_RUNTIME_ALREADY_IMPORTED:'+name)
    os.environ['SEARCH_G1_LEGACY_SOURCE_ROOT']=str(root)
    return root


def setup_worker():
    """Ray worker-process hook runs before actor/function deserialization."""
    activate()
