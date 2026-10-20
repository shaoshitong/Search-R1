"""Resolve only the two NAS10 spellings; preserve proof bytes and stored paths."""
import hashlib
import inspect
import json
from pathlib import Path
import re


def readable(path, base, physical):
    path, base, physical = Path(path), Path(base), Path(physical)
    relative = None
    for root in (base, physical):
        if path.is_relative_to(root):
            assert '..' not in path.parts
            relative = path.relative_to(root)
            break
    if relative is None:return path
    alias = base / relative
    # Select by visibility only. Never retry a different file after SHA failure.
    return alias if alias.exists() else physical / relative


def read(path, base, physical):
    return json.loads(readable(path, base, physical).read_text())


def sha(path, base, physical):
    return hashlib.sha256(readable(path, base, physical).read_bytes()).hexdigest()


def remote_reads(source):
    """Apply the same resolver before a remote control verifies its stage.

The generated stdin program cannot import a staged helper before locating the
stage, so embed this exact function's source; its file is also in the manifest.
"""
    pattern = r'^def read\(p\):return json.loads\(Path\(p\).read_text\(\)\)(if Path\(p\).exists\(\)else None)?\n'
    match = re.search(pattern, source, re.MULTILINE);assert match
    optional = bool(match.group(1))
    source, count = re.subn(pattern, '', source, count=1, flags=re.MULTILINE);assert count == 1
    old = 'def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()\n'
    assert source.count(old) == 1;source = source.replace(old, '')
    helper = inspect.getsource(readable) + '''
def read(p):
 path=readable(p,BASE,PHYSICAL)
 return json.loads(path.read_text())''' + ('if path.exists()else None' if optional else '') + '''
def sha(p):return hashlib.sha256(readable(p,BASE,PHYSICAL).read_bytes()).hexdigest()
'''
    marker = 'from pathlib import Path\n';assert source.count(marker) == 1
    source = source.replace(marker, marker + helper, 1)
    source, count = re.subn(r'base=Path\((PHYSICAL if ACTION[^\n]+?else BASE)\);root=',
        r'base=readable(\1,BASE,PHYSICAL);root=', source, count=1)
    assert count == 1
    return source
