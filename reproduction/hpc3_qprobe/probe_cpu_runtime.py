"""Pinned, private CPU probe dependencies; never change reused model runtime."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
LOCK=Path(__file__).with_name('probe_cpu_wheels.json')


def lock():return json.loads(LOCK.read_text())


def root():
    fingerprint=hashlib.sha256(json.dumps(lock(),sort_keys=True).encode()).hexdigest()
    return ROOT/'runtime'/('probe-cpu-'+fingerprint[:16])


def verify():
    target=root();manifest=json.loads((target/'manifest.json').read_text())
    if manifest['wheels']!=lock() or manifest['status']!='CPU_PROBE_WHEELS_FULL_SHA_PASS':
        raise ValueError('Required CPU probe dependency binding differs')
    for n,h in manifest['files'].items():
        p=target/'site-packages'/n
        if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:
            raise ValueError('Private CPU probe dependency changed: '+n)
    return target/'site-packages',manifest


def fit_gate():
    value=json.loads((root()/'actual-fit.json').read_text())
    expected=dict(joblib='1.4.2',sklearn='1.5.2',scipy='1.13.1',threadpoolctl='3.5.0',numpy='1.26.4')
    if value['status']!='CPU_PROBE_ACTUAL_FIT_AND_ROUNDTRIP_PASS' or value['versions']!=expected:
        raise ValueError('Private CPU probe actual fitted runtime not accepted')
    if value['site']!=str(root()/'site-packages') or value['actual_gpu_allocation'] is not False or value['actual_training_updates']!=0:
        raise ValueError('Private CPU fitted runtime proof scope differs')
    return value


def fit():
    """Real finite balanced-LR fit and exact joblib round trip, with no GPU."""
    import io,time
    import numpy as np,joblib,sklearn,scipy,threadpoolctl
    from sklearn.linear_model import LogisticRegression
    site,proof=verify()
    x=np.array([[0,1],[1,0],[0,2],[2,0]],dtype=np.float32);y=np.array([0,1,0,1])
    clf=LogisticRegression(class_weight='balanced',C=1,max_iter=4000).fit(x,y)
    a=clf.predict_proba(x);stream=io.BytesIO();joblib.dump(clf,stream);stream.seek(0)
    b=joblib.load(stream).predict_proba(x)
    assert np.isfinite(a).all() and np.array_equal(a,b) and np.array_equal(clf.predict(x),y)
    value=dict(status='CPU_PROBE_ACTUAL_FIT_AND_ROUNDTRIP_PASS',epoch=time.time(),
        versions=dict(joblib=joblib.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,threadpoolctl=threadpoolctl.__version__,numpy=np.__version__),
        site=str(site),file_count=len(proof['files']),actual_gpu_allocation=False,actual_training_updates=0)
    receipt=root()/'actual-fit.json'
    if receipt.exists():fit_gate()
    else:
        with receipt.open('x') as f:json.dump(value,f,indent=2)
    return value


if __name__=='__main__':print(json.dumps(fit()),flush=True)
