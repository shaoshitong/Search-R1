"""Verify a pushed Git tree and build the small, raw-data-free V7 deliverables."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

repo = Path(__file__).resolve().parents[2]
output = repo.parents[1] / "outputs"
output.mkdir(exist_ok=True)
branch = "codex/ai-creator-game-20261104"

def git(*args):
    return subprocess.check_output(["git", "-C", str(repo), *args])

subprocess.run(["git", "-C", str(repo), "fetch", "--no-tags", "origin", branch], check=True)
commit = git("rev-parse", "HEAD").decode().strip()
fetched = git("rev-parse", "FETCH_HEAD").decode().strip()
remote = git("ls-remote", "origin", "refs/heads/" + branch).decode().split()[0]
assert commit == fetched == remote
names = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
files = [name for name in names if name.startswith("src/v7/") or name.startswith("docs/V7")
         or name == "docs/AI_CREATOR_GAME_STATE_MODEL_ZH_V7.pdf"
         or name in [".gitattributes", ".gitignore"]]
manifest = []
for name in files:
    payload = (repo / name).read_bytes()
    blob = git("show", f"{fetched}:{name}")
    sha = hashlib.sha256(payload).hexdigest()
    assert sha == hashlib.sha256(blob).hexdigest(), name
    manifest.append(dict(path=name, bytes=len(blob), sha256=sha))
publication = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
                   repository="git@github.com:shaoshitong/Search-R1.git", branch=branch,
                   source_commit=commit, remote_commit=remote, fresh_fetch_commit=fetched,
                   git_readback_manifest_matches=True, files=manifest,
                   force_push=False, raw_data_in_source_archive=False)
(output / "V7_CODE_PUBLICATION.json").write_text(json.dumps(publication, indent=2) + "\n")
pdf = output / "AI_CREATOR_GAME_STATE_MODEL_ZH_V7.pdf"
shutil.copy2(repo / "docs/AI_CREATOR_GAME_STATE_MODEL_ZH_V7.pdf", pdf)
archive = output / "AI_CREATOR_GAME_STATE_MODEL_ZH_V7_SOURCE.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
    for name in files:
        bundle.write(repo / name, name)
    bundle.write(repo / "src/v7/README.md", "README.md")
    bundle.write(output / "V7_CODE_PUBLICATION.json", "CODE_PUBLICATION.json")
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for row in manifest:
        assert hashlib.sha256(bundle.read(row["path"])).hexdigest() == row["sha256"]
receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_commit=commit,
               source_files_verified=len(files),
               pdf=dict(path=str(pdf), bytes=pdf.stat().st_size,
                        sha256=hashlib.sha256(pdf.read_bytes()).hexdigest()),
               source_zip=dict(path=str(archive), bytes=archive.stat().st_size,
                               sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),
               zip_entries_verified_against_git=len(files), remote_published=True)
(output / "V7_DELIVERY_ACCEPTANCE.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
