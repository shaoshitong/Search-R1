"""Read small round-zero fitting evidence; no model/rollout loading or mutation."""
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value, key=""):
    if isinstance(value, dict):
        return {k: compact(v, k) for k, v in value.items()}
    if isinstance(value, list) and len(value) > 32:
        return {"count": len(value), "json_sha256": hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()).hexdigest()}
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, str) and len(value) > 1000:
        return {"characters": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest()}
    return value


def collect(manifests):
    result = {"observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "training_modified": False, "model_loaded": False, "manifests": []}
    for filename in manifests:
        path = Path(filename)
        payload = json.loads(path.read_text())
        record = {"path": str(path), "actual_sha256": sha(path),
                  "manifest": compact(payload), "related_small_files": {}, "head_files": {}}
        for name in ("calibration.log", "heads/calibration-report.json", "dumps/calibration-label-proof.json"):
            small = path.parent / name
            if small.is_file():
                entry = {"bytes": small.stat().st_size, "sha256": sha(small)}
                if name.endswith(".json"):
                    entry["content"] = compact(json.loads(small.read_text()))
                else:
                    text = small.read_text(errors="replace")
                    entry["fit_status_lines"] = [line[:1500] for line in text.splitlines()
                        if any(token in line.lower() for token in
                               ("saved", "auc", "fitted", "class_counts", "calibration-report"))][-20:]
                record["related_small_files"][name] = entry
        for role in ("commit", "boundary"):
            head = Path(payload[role])
            record["head_files"][role] = {"path": str(head), "exists": head.is_file(),
                "bytes": head.stat().st_size if head.is_file() else None,
                "actual_sha256": sha(head) if head.is_file() else None}
        result["manifests"].append(record)
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(collect(sys.argv[1:]), ensure_ascii=False))
