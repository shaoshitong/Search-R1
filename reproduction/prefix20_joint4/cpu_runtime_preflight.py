"""Real CPU imports/fit, durable before GPU submission; no model/GPU work."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time


def execute(output):
    from reproduction.hpc3_qprobe.probe_cpu_runtime import root,lock,fit_gate
    output=Path(output)
    if not output.is_absolute() or output.resolve()!=output or output.exists():
        raise ValueError('PREFIX20_CPU_PREFLIGHT_EXPLICIT_NEW_OUTPUT_REQUIRED')
    claim=output.with_name(output.name+'.claim.json')
    with claim.open('x') as stream:
        json.dump(dict(pid=os.getpid(),uid=os.getuid(),started_unix=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),stream)
        stream.flush();os.fsync(stream.fileno())
    site=root()/'site-packages';manifest=json.loads((root()/'manifest.json').read_text())
    if manifest['wheels']!=lock():raise ValueError('PREFIX20_CPU_PREFLIGHT_WHEEL_LOCK_DIFFERS')
    historical=fit_gate();sys.path.insert(0,str(site))
    import numpy as np,joblib,sklearn,scipy,threadpoolctl
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score,accuracy_score
    modules=(joblib,sklearn,scipy,threadpoolctl)
    files={}
    for module in modules:
        path=Path(module.__file__).resolve();relative=str(path.relative_to(site))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest['files'].get(relative)!=digest:raise ValueError('PREFIX20_CPU_PREFLIGHT_IMPORTED_MODULE_SHA_DIFFERS')
        files[module.__name__]=dict(path=str(path),sha256=digest)
    versions={m.__name__:m.__version__ for m in (*modules,np)}
    if versions!=historical['versions']:raise ValueError('PREFIX20_CPU_PREFLIGHT_ACTUAL_VERSION_DIFFERS')
    x=np.array([[0,1],[1,0],[0,2],[2,0]],dtype=np.float32);y=np.array([0,1,0,1])
    model=LogisticRegression(class_weight='balanced',C=1,max_iter=4000).fit(x,y)
    first=model.predict_proba(x);buffer=io.BytesIO();joblib.dump(model,buffer);buffer.seek(0)
    second=joblib.load(buffer).predict_proba(x)
    if not np.isfinite(first).all() or not np.array_equal(first,second) or not np.array_equal(model.predict(x),y):
        raise ValueError('PREFIX20_CPU_PREFLIGHT_REAL_FIT_OR_ROUNDTRIP_FAILED')
    value=dict(status='ACTUAL_PINNED_CPU_IMPORT_FIT_ROUNDTRIP_PASS',versions=versions,module_files=files,
        numpy_file=np.__file__,actual_gpu_allocation=False,actual_training_updates=0,torch_imported='torch' in sys.modules,
        full_vendor_sha_reverified=False,full_vendor_sha_scope='Fresh full SHA remains bootstrap verify; this proof checks actual imported modules and CPU execution',
        checked_unix=time.time())
    with output.open('x') as stream:json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(execute(args.output)),flush=True)
