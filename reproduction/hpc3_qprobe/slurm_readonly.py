"""Read-only scheduler transport failures are unknown, never job outcomes."""
import subprocess
import time


class SlurmReadUnavailable(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__('scheduler query unavailable: '+repr(evidence))


def query(argv, *, timeout=30):
    argv = list(argv)
    if not argv or (argv[0] not in ('sacct', 'squeue') and
                    not (argv[:3] == ['scontrol', 'show', 'job'])):
        raise ValueError('Only read-only scheduler queries may use this retry classification')
    started = time.time()
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise SlurmReadUnavailable(dict(argv=argv, started_unix=started, checked_unix=time.time(),
            returncode=None, error_type=type(exc).__name__, error=str(exc), state='UNKNOWN')) from exc
    if result.returncode:
        raise SlurmReadUnavailable(dict(argv=argv, started_unix=started, checked_unix=time.time(),
            returncode=result.returncode, stdout=result.stdout[-16384:], stderr=result.stderr[-16384:],
            state='UNKNOWN'))
    return result.stdout
