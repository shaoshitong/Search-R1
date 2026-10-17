"""Bounded direct controls and code-only bootstrap transport; no model relay."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex

from .direct_control import run as direct


def adapter(source_address):
    def run(command, payload=b'', account='mou176'):
        def call(script, timeout=60):
            return direct(script, account=account, source_address=source_address, timeout=timeout)
        if not payload:
            return call(command)
        # Only the publisher's authenticated code archive is supported. Never
        # make this adapter a general binary/data transport fallback.
        header, archive = payload.split(b'\n', 1)
        record = json.loads(header)
        base = Path(record['plan']['base'])
        from .common import DEFAULT_BASE
        assert re.fullmatch(re.escape(str(DEFAULT_BASE))+r'(?:-r[1-9][0-9]*)?', str(base))
        assert record['publication']['pushed_and_fetched'] is True
        assert hashlib.sha256(archive).hexdigest() == record['archive_sha256']
        from .control import CONTROL_PYTHON, REPO
        assert shlex.split(command) == [CONTROL_PYTHON, '-c',
            (REPO/'reproduction/hpc3_dynrpo_periodic/bootstrap.py').read_text()]
        digest = hashlib.sha256(payload).hexdigest()
        target = base/'preparation'/('code-only-upload-'+digest)
        for offset in range(0, len(payload), 24000):
            data = base64.b64encode(payload[offset:offset+24000]).decode()
            script = ('import os,base64; from pathlib import Path; p=Path('+repr(str(target))+'); '
                'p.parent.mkdir(parents=True,exist_ok=True); '
                'assert p.parent.parent.stat().st_uid==os.getuid(); '
                'assert not p.is_symlink(); d=base64.b64decode('+repr(data)+'); '
                'n='+str(offset)+'; size=p.stat().st_size if p.exists() else 0; '
                'assert size==n or (size>=n+len(d) and p.read_bytes()[n:n+len(d)]==d); '
                'f=p.open("ab"); f.write(d) if size==n else None; f.close()')
            result = call(CONTROL_PYTHON+' -c '+shlex.quote(script))
            if result.returncode:
                return result
        script = ('import hashlib,subprocess; from pathlib import Path; '
            'p=Path('+repr(str(target))+'); data=p.read_bytes(); '
            'assert hashlib.sha256(data).hexdigest()=='+repr(digest)+'; '
            'raise SystemExit(subprocess.run('+repr(shlex.split(command))+',input=data).returncode)')
        return call(CONTROL_PYTHON+' -c '+shlex.quote(script), timeout=60)
    return run
