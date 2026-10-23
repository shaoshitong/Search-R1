"""Runtime source binding includes the two extensionless package version files."""
from pathlib import Path

VERSION_FILES = ('verl/version/version', 'verl/single_controller/version/version')
SUFFIXES = ('.py', '.sh', '.yaml', '.yml', '.json')


def runtime_files(repo):
    """Include independent recipe modules and launchers in the normal SHA map."""
    repo = Path(repo)
    for name in VERSION_FILES:
        assert (repo / name).is_file(), 'Incomplete runtime package: ' + name
    return sorted(p for p in repo.rglob('*')
                  if p.is_file() and (p.suffix in SUFFIXES or str(p.relative_to(repo)) in VERSION_FILES)
                  and not {'.git', '__pycache__', 'evidence'}.intersection(p.relative_to(repo).parts))
