"""Pinned official 7B bytes; no teacher/probe checkpoint is an actor source."""
import json
from pathlib import Path

REPO = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
FORMAT = 'hpc3-official-qwen25-7b-a09a354-eleven-files-v1'
MODEL = ('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/'
         'assets/qwen25-7b-instruct-a09a35458c70')


def expected():
    return json.loads(Path(__file__).with_name('official7b.json').read_text())


def verify(model, manifest):
    from reproduction.common8.verify_data import digest
    model = Path(model)
    reference = expected()
    if manifest != reference:
        raise ValueError('EASY7B_OFFICIAL_MANIFEST_DIFFERS')
    hashes = {}
    for item in reference['files']:
        p = model / item['file']
        if p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            raise ValueError('EASY7B_OFFICIAL_BYTES_DIFFER: ' + item['file'])
        hashes[item['file']] = item['sha256']
    config = json.loads((model / 'config.json').read_text())
    if (config['model_type'], config['hidden_size'], config['num_hidden_layers']) != ('qwen2', 3584, 28):
        raise ValueError('EASY7B_OFFICIAL_ARCHITECTURE_DIFFERS')
    return hashes


def create(model, path):
    from .control import publish
    reference = expected()
    verify(model, reference)
    publish(path, reference)
    return reference
