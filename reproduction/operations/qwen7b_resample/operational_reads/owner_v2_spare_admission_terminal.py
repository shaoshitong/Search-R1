"""Seal seven exact real capacity failures, partitioned by their actual node.

Plan is local/static. Seal writes only its shared once-proof; it never signals,
starts, stops, loads a model, reads inputs/trajectories or polls healthy units.
Children were omitted by the original collector: capture their original file
at seal time, report absent historical births honestly, and require numeric PID
absence, empty own cgroups and no own NVML contexts twice at least 3s apart.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import time

BASE = Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
ALIAS = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
RUN = '70188b8e50fd4b45bc420ce58014f46e'
STATUS = 'OWN_OWNER_V2_SPARE_ADMISSION_CAPACITY_FAILURE_TERMINAL_FULL_SHA_RELEASE_PASS'
SPECS = [{'node': 'h100-5',
  'key': 'source:gpu1-tp1',
  'unit': 'sst-q7b-g5-guard-rollover-owner-v2-gpu1-tp1-r1-70188b8e50fd.service',
  'pid': 2725517,
  'job': 'pool/jobs/job-source-slot-gpu1-tp1-2725517/attempt-1792544323119306519-2725517',
  'allocation_sha256': 'ff875797b5322ac0d342701faafb21966a4a97db2511ec3133cf191da81fe6f6',
  'failure_sha256': 'dade59d5b56f732d6eecedbc6115b1a5a3787d1766daa2e0407f567c2e8a2f8e',
  'pins': [('pool/jobs/job-source-slot-gpu1-tp1-2725517/attempt-1792544323119306519-2725517/sampler-launches/launch-00/failure.json',
            '67e6112b958326dbfa2ab90e4d54941df7cd6d363212ccb3082128751c9b3caa'),
           ('pool/jobs/job-source-slot-gpu1-tp1-2725517/attempt-1792544323119306519-2725517/sampler-launches/launch-01/failure.json',
            'ce523cf1c95a922a3a9a5fec14ee6da4c5352bfad27fe5872a1c7ac031423adb'),
           ('pool/jobs/job-source-slot-gpu1-tp1-2725517/attempt-1792544323119306519-2725517/sampler-launches/launch-02/failure.json',
            '7289de0e3bcaa71e0c8cc697ad4a03268346ac83744ec4b87af6953630716353'),
           ('retrieval-owner-generation/launch/native-2725517/allocation-evidence.json',
            '5b8dd8fd034804b554b16688e55038b93c8d6b85c0f2d680b1f57eaa22fa601d')],
  'native': 'retrieval-owner-generation/launch/native-2725517',
  'stage': 'preparation/retrieval-generation-guard-rollover-owner-v2-gpu1-tp1-r1/f4d72e4f9d4b89e2b13a638b8b530dd075ea12e5121338157f8e202a0c45446f',
  'snapshot': {'summary_path': '/tmp/q7b-guarded-after-all14-once.json',
               'summary_sha256': '4f7ea786966b13d6b88d22594e50a8669f914d49c411ea2910c7e025e85f318d',
               'original_path': '/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qwen7b-guarded-follow-9f17687c103eb2ac/actual-h100-5-1792544553207090000.json',
               'original_sha256': '05e3d85d7e52a1dbed7f5913a2eaf135607c94d9cbdbc8a02028a625398eb677',
               'native': {'MainPID': '0',
                          'Result': 'exit-code',
                          'ExecMainStatus': '1',
                          'ControlGroup': '',
                          'Id': 'sst-q7b-g5-guard-rollover-owner-v2-gpu1-tp1-r1-70188b8e50fd.service',
                          'LoadState': 'loaded',
                          'ActiveState': 'failed',
                          'SubState': 'failed'}}},
 {'node': 'h100-5',
  'key': 'source:gpu3-tp1',
  'unit': 'sst-q7b-g5-guard-rollover-owner-v2-gpu3-tp1-r1-70188b8e50fd.service',
  'pid': 2726199,
  'job': 'pool/jobs/job-source-slot-gpu3-tp1-2726199/attempt-1792544343759366679-2726199',
  'allocation_sha256': '76ee7462a736bcda780720e1067185a2cf4ed4157936278b6ff74e8d04a14f20',
  'failure_sha256': 'f1d3c6909d4e210aa3a4ed60f84b05acb8dde313e920ea173060c2e153b756d2',
  'pins': [('pool/jobs/job-source-slot-gpu3-tp1-2726199/attempt-1792544343759366679-2726199/sampler-launches/launch-00/failure.json',
            '88da89ef69ec014f77944cf8f1c44734dc16ba1f9106143eec707541221153d5'),
           ('pool/jobs/job-source-slot-gpu3-tp1-2726199/attempt-1792544343759366679-2726199/sampler-launches/launch-01/failure.json',
            '9aa9d27fdb7231c42ff93f84826e6efe0bb1c3a580793ad1fe9f731fad2b6698'),
           ('pool/jobs/job-source-slot-gpu3-tp1-2726199/attempt-1792544343759366679-2726199/sampler-launches/launch-02/failure.json',
            '1d349b94eac57cc5c749ac8dda20a7f21aff3425a11567510df8f23c0c626bfc'),
           ('retrieval-owner-generation/launch/native-2726199/allocation-evidence.json',
            '9a55215d07756044158f0aa0f4d2781e70f2b39a26d55218f05b358cc4bd4210')],
  'native': 'retrieval-owner-generation/launch/native-2726199',
  'stage': 'preparation/retrieval-generation-guard-rollover-owner-v2-gpu3-tp1-r1/9691472f219c5081e28505511c32ff5e956b1180aa8c7ede415ab38b53c04843',
  'snapshot': {'summary_path': '/tmp/q7b-guarded-after-all14-once.json',
               'summary_sha256': '4f7ea786966b13d6b88d22594e50a8669f914d49c411ea2910c7e025e85f318d',
               'original_path': '/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qwen7b-guarded-follow-9f17687c103eb2ac/actual-h100-5-1792544553207090000.json',
               'original_sha256': '05e3d85d7e52a1dbed7f5913a2eaf135607c94d9cbdbc8a02028a625398eb677',
               'native': {'MainPID': '0',
                          'Result': 'exit-code',
                          'ExecMainStatus': '1',
                          'ControlGroup': '',
                          'Id': 'sst-q7b-g5-guard-rollover-owner-v2-gpu3-tp1-r1-70188b8e50fd.service',
                          'LoadState': 'loaded',
                          'ActiveState': 'failed',
                          'SubState': 'failed'}}},
 {'node': 'h100-5',
  'key': 'source:gpu4-tp1',
  'unit': 'sst-q7b-g5-guard-rollover-owner-v2-gpu4-tp1-r1-70188b8e50fd.service',
  'pid': 2726640,
  'job': 'pool/jobs/job-source-slot-gpu4-tp1-2726640/attempt-1792544353853687457-2726640',
  'allocation_sha256': '602847876e2998117ef6b470670d50b32e24ecf94fea9f6d22d5229debb904ef',
  'failure_sha256': '92cc72b7d4e76e006dc5ebf0007f0ebe2a35cd1b7ca799332b3bae01d122359e',
  'pins': [('pool/jobs/job-source-slot-gpu4-tp1-2726640/attempt-1792544353853687457-2726640/sampler-launches/launch-00/failure.json',
            'c9a9e3717a64a69356f59b66e950ee424502d041263b55538c4ec1bcaffe983e'),
           ('pool/jobs/job-source-slot-gpu4-tp1-2726640/attempt-1792544353853687457-2726640/sampler-launches/launch-01/failure.json',
            'b872e75fb4da058198c91f4d60dd2741d07cc16fac8df2f0e06152b7a3592788'),
           ('pool/jobs/job-source-slot-gpu4-tp1-2726640/attempt-1792544353853687457-2726640/sampler-launches/launch-02/failure.json',
            '0978b159848cefefe5dcb658a6c3207fae77160dd9232b5746c1206f8f655198'),
           ('retrieval-owner-generation/launch/native-2726640/allocation-evidence.json',
            '9c90e499af16fa7548779df38e1ef58a60c1d876e1bbd9e752319c7d04566def')],
  'native': 'retrieval-owner-generation/launch/native-2726640',
  'stage': 'preparation/retrieval-generation-guard-rollover-owner-v2-gpu4-tp1-r1/455a59f4349149c518d6cc584095c0698666bc53fa2e84420a03a90685cf8097',
  'snapshot': {'summary_path': '/tmp/q7b-guarded-after-all14-once.json',
               'summary_sha256': '4f7ea786966b13d6b88d22594e50a8669f914d49c411ea2910c7e025e85f318d',
               'original_path': '/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qwen7b-guarded-follow-9f17687c103eb2ac/actual-h100-5-1792544553207090000.json',
               'original_sha256': '05e3d85d7e52a1dbed7f5913a2eaf135607c94d9cbdbc8a02028a625398eb677',
               'native': {'MainPID': '0',
                          'Result': 'exit-code',
                          'ExecMainStatus': '1',
                          'ControlGroup': '',
                          'Id': 'sst-q7b-g5-guard-rollover-owner-v2-gpu4-tp1-r1-70188b8e50fd.service',
                          'LoadState': 'loaded',
                          'ActiveState': 'failed',
                          'SubState': 'failed'}}},
 {'node': 'h100-5',
  'key': 'source:gpu6-tp1',
  'unit': 'sst-q7b-g5-guard-rollover-owner-v2-gpu6-tp1-r1-70188b8e50fd.service',
  'pid': 2727057,
  'job': 'pool/jobs/job-source-slot-gpu6-tp1-2727057/attempt-1792544364191728793-2727057',
  'allocation_sha256': '3d2b8f0aa69ad411cbad8fd05cc383a48088ab1e5cda6cf953c7c874880c349f',
  'failure_sha256': 'aad0283f756ee15284e7719d890096827cf28b60c4702edc9daaf7a5f14b4996',
  'pins': [('pool/jobs/job-source-slot-gpu6-tp1-2727057/attempt-1792544364191728793-2727057/sampler-launches/launch-00/failure.json',
            '924b53c5c5b9aa91b868f9a5c914b39ee107f5ca53e505ef7374ceba290e66f7'),
           ('pool/jobs/job-source-slot-gpu6-tp1-2727057/attempt-1792544364191728793-2727057/sampler-launches/launch-01/failure.json',
            'a38396f11a56f33f14aec892e91e341d45bd85a0d202a099873d9ffcbcd30d9d'),
           ('pool/jobs/job-source-slot-gpu6-tp1-2727057/attempt-1792544364191728793-2727057/sampler-launches/launch-02/failure.json',
            '9aae077b45f1d9eb73520ae789787d982bd039bd59ccf7a4b80862fcab6d424e'),
           ('retrieval-owner-generation/launch/native-2727057/allocation-evidence.json',
            '14e76499e6b015c7271f6b41b6040cd7c52e61acb5fab6e977ed004d1c364b24')],
  'native': 'retrieval-owner-generation/launch/native-2727057',
  'stage': 'preparation/retrieval-generation-guard-rollover-owner-v2-gpu6-tp1-r1/e729d32bc72e9ce1ae48a32747f072179b1f0349d0c265ec23b488a8f8cf91e7',
  'snapshot': {'summary_path': '/tmp/q7b-guarded-after-all14-once.json',
               'summary_sha256': '4f7ea786966b13d6b88d22594e50a8669f914d49c411ea2910c7e025e85f318d',
               'original_path': '/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qwen7b-guarded-follow-9f17687c103eb2ac/actual-h100-5-1792544553207090000.json',
               'original_sha256': '05e3d85d7e52a1dbed7f5913a2eaf135607c94d9cbdbc8a02028a625398eb677',
               'native': {'MainPID': '0',
                          'Result': 'exit-code',
                          'ExecMainStatus': '1',
                          'ControlGroup': '',
                          'Id': 'sst-q7b-g5-guard-rollover-owner-v2-gpu6-tp1-r1-70188b8e50fd.service',
                          'LoadState': 'loaded',
                          'ActiveState': 'failed',
                          'SubState': 'failed'}}},
 {'node': 'h100-5',
  'key': 'source:gpu7-tp1',
  'unit': 'sst-q7b-g5-guard-rollover-owner-v2-gpu7-tp1-r1-70188b8e50fd.service',
  'pid': 2727678,
  'job': 'pool/jobs/job-source-slot-gpu7-tp1-2727678/attempt-1792544374744608247-2727678',
  'allocation_sha256': '8782bb624404f5b5b72b3d68da7df987fa1f754c3f7aabec8b5b23c77dde407f',
  'failure_sha256': '780a1ec9d6fadaa3edc1feb26c9cc0881e36108d46b1830da118c83ede294dc7',
  'pins': [('pool/jobs/job-source-slot-gpu7-tp1-2727678/attempt-1792544374744608247-2727678/sampler-launches/launch-00/failure.json',
            '56cecc8bc79967f28c981ce88ce80cf31e9703ecc68b819b8bfa7603fce91efd'),
           ('pool/jobs/job-source-slot-gpu7-tp1-2727678/attempt-1792544374744608247-2727678/sampler-launches/launch-01/failure.json',
            'b39df1e1064f4869c9d4fdc7b85741a24aa9f3734cea3def1adda1505f3a3abf'),
           ('pool/jobs/job-source-slot-gpu7-tp1-2727678/attempt-1792544374744608247-2727678/sampler-launches/launch-02/failure.json',
            'a96b2905c4e4c23ef37dd44fe3ac12918ca25c8f75b76158b9e7694c077f2d46'),
           ('retrieval-owner-generation/launch/native-2727678/allocation-evidence.json',
            '5db8a352a15ec7739c5716327bf575c58157773964a7bc189a0d337f466f36a6')],
  'native': 'retrieval-owner-generation/launch/native-2727678',
  'stage': 'preparation/retrieval-generation-guard-rollover-owner-v2-gpu7-tp1-r1/1940e998467a787048d11fb5248df78d576c20e00ce2ba98a596576a08f5bb06',
  'snapshot': {'summary_path': '/tmp/q7b-guarded-after-all14-once.json',
               'summary_sha256': '4f7ea786966b13d6b88d22594e50a8669f914d49c411ea2910c7e025e85f318d',
               'original_path': '/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qwen7b-guarded-follow-9f17687c103eb2ac/actual-h100-5-1792544553207090000.json',
               'original_sha256': '05e3d85d7e52a1dbed7f5913a2eaf135607c94d9cbdbc8a02028a625398eb677',
               'native': {'MainPID': '0',
                          'Result': 'exit-code',
                          'ExecMainStatus': '1',
                          'ControlGroup': '',
                          'Id': 'sst-q7b-g5-guard-rollover-owner-v2-gpu7-tp1-r1-70188b8e50fd.service',
                          'LoadState': 'loaded',
                          'ActiveState': 'failed',
                          'SubState': 'failed'}}},
 {'node': 'h100-10',
  'key': 'node10:gpu1-tp1',
  'unit': 'sst-q7b-g5-relaygen-queued-owner-v2-slot-gpu1-tp1-r1-70188b8e50fd.service',
  'pid': 3224037,
  'job': 'pool/jobs/job-relay-slot-gpu1-tp1-3224037/attempt-1792544353953793814-3224037',
  'allocation_sha256': '105e63b5e4d6627be8244a2cc1d5269337a34150017dac4061bb153da94141fd',
  'failure_sha256': '9874eeaee05593d03b4a8a96639e100204aa4a6f743c4a6b95e385f5b4ca013b',
  'pins': [('pool/jobs/job-relay-slot-gpu1-tp1-3224037/attempt-1792544353953793814-3224037/sampler-launches/launch-00/failure.json',
            '7529e586772de9086ca8944a41e43d7c9b16f86aef3273a6c2c7633f83eb1306'),
           ('retrieval-relay-generation/launch/native-3224037/allocation-evidence.json',
            '91ab8f31604b584dbc0376e62095cefbfe9f854cac072304e047516fe744505c')],
  'native': 'retrieval-relay-generation/launch/native-3224037',
  'stage': 'preparation/relay-generation-queued-owner-v2-slot-gpu1-tp1-r1/6989939c9c0b103961714ed063ce5fb45c45f4379faa401fbc5ffc7f0cf0b58c',
  'snapshot': {'summary_path': '/tmp/q7b-guarded-after-all14-once.json',
               'summary_sha256': '4f7ea786966b13d6b88d22594e50a8669f914d49c411ea2910c7e025e85f318d',
               'original_path': '/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qwen7b-guarded-follow-9f17687c103eb2ac/actual-h100-10-1792544558265288000.json',
               'original_sha256': 'c3dcf9bc51ac7f38014f0d64fd52df7d5f2a4736af1a01a7a914a804f6a4ccd4',
               'native': {'MainPID': '0',
                          'Result': 'exit-code',
                          'ExecMainStatus': '1',
                          'ControlGroup': '',
                          'Id': 'sst-q7b-g5-relaygen-queued-owner-v2-slot-gpu1-tp1-r1-70188b8e50fd.service',
                          'LoadState': 'loaded',
                          'ActiveState': 'failed',
                          'SubState': 'failed'}}},
 {'node': 'h100-10',
  'key': 'node10:gpu2-tp1',
  'unit': 'sst-q7b-g5-relaygen-queued-owner-v2-slot-gpu2-tp1-r1-70188b8e50fd.service',
  'pid': 3224468,
  'job': 'pool/jobs/job-relay-slot-gpu2-tp1-3224468/attempt-1792544363325779504-3224468',
  'allocation_sha256': '110daebab1e099ae829dd18e47a22e3a1d38f35d61eba620f95e2834203a693f',
  'failure_sha256': 'e2f3e396db20e7c4fd67e178638ce4ba7b757c7f6a22603b3f11edfaa47f3164',
  'pins': [('pool/jobs/job-relay-slot-gpu2-tp1-3224468/attempt-1792544363325779504-3224468/sampler-launches/launch-00/failure.json',
            '723936d2b9289faca0bf6af733e9738327c3d3fef304fd5a803c25ed2bba7dc9'),
           ('retrieval-relay-generation/launch/native-3224468/allocation-evidence.json',
            'feab68c2ce353cf6011bbdfaf78f477577f9d561f221924e5c9a42a00d8ccef6')],
  'native': 'retrieval-relay-generation/launch/native-3224468',
  'stage': 'preparation/relay-generation-queued-owner-v2-slot-gpu2-tp1-r1/f06b48e3ee5146857254e46dc8ebd585f96a15c05ed79b72aadd59b96d1c4b66',
  'snapshot': {'summary_path': '/tmp/q7b-guarded-after-all14-once.json',
               'summary_sha256': '4f7ea786966b13d6b88d22594e50a8669f914d49c411ea2910c7e025e85f318d',
               'original_path': '/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/qwen7b-guarded-follow-9f17687c103eb2ac/actual-h100-10-1792544558265288000.json',
               'original_sha256': 'c3dcf9bc51ac7f38014f0d64fd52df7d5f2a4736af1a01a7a914a804f6a4ccd4',
               'native': {'MainPID': '0',
                          'Result': 'exit-code',
                          'ExecMainStatus': '1',
                          'ControlGroup': '',
                          'Id': 'sst-q7b-g5-relaygen-queued-owner-v2-slot-gpu2-tp1-r1-70188b8e50fd.service',
                          'LoadState': 'loaded',
                          'ActiveState': 'failed',
                          'SubState': 'failed'}}}]


def selected(node):
    rows = [row for row in SPECS if row['node'] == node]
    assert len(rows) == (5 if node == 'h100-5' else 2) and len({row['unit'] for row in rows}) == len(rows)
    return rows


class View:
    def __init__(self, base):
        self.base = Path(base);assert self.base in (BASE, ALIAS)
    def relative(self, path):
        path = Path(path);assert '..' not in path.parts
        for root in (BASE, ALIAS):
            if path.is_relative_to(root):return path.relative_to(root)
        raise ValueError('Path leaves this same-run NAS root')
    def path(self, path):return self.base / self.relative(path)
    def canonical(self, path):return str(BASE / self.relative(path))


def read(path):return json.loads(Path(path).read_text())


def identity(value):return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def original(view, path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno());assert stat.S_ISREG(before.st_mode) and identity(before) == identity(path.stat())
        digest = hashlib.sha256()
        for raw in iter(lambda:stream.read(1024 * 1024), b''):digest.update(raw)
        assert identity(before) == identity(os.fstat(stream.fileno())) == identity(path.stat())
    return dict(path=view.canonical(path), bytes=before.st_size, sha256=digest.hexdigest(), device=before.st_dev,
        inode=before.st_ino, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns)


def stable(view, records):
    for row in records:
        assert identity(view.path(row['path']).stat()) == tuple(row[key] for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'))


def metadata(roots):
    found = set()
    for root in roots:
        assert root.is_dir() and not root.is_symlink()
        for path in root.rglob('*'):
            if any(part in ('cache', 'sampling', 'features', 'states', 'retrieval-cache', 'tmp', 'inputs', 'assets', '__pycache__') for part in path.relative_to(root).parts):continue
            if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.log', '.txt', '.err', '.out', '.py', '.sh'):
                if not any(word in path.name.lower() for word in ('secret', 'token')):found.add(path)
    return found


def prepare(view, node):
    rows, records, roots, paths, known = [], {}, [], set(), {}
    for spec in selected(node):
        job = view.path(BASE / spec['job']);assert job.parent.parent == view.base / 'pool/jobs'
        expected = dict(spec['pins']);expected['%s/allocation.json' % spec['job']] = spec['allocation_sha256']
        expected['%s/failure.json' % spec['job']] = spec['failure_sha256']
        for relative, digest in expected.items():
            path = view.path(BASE / relative);row = original(view, path)
            assert row['sha256'] == digest, 'Pinned original differs: ' + relative
            records[path] = row
        allocation = read(job / 'allocation.json');failure = read(job / 'failure.json')
        assert allocation['run_id'] == RUN and allocation['node'] == node and allocation['native_unit'] == spec['unit']
        assert allocation['controller_pid'] == spec['pid'] and allocation['native_cgroup'] == '/system.slice/' + spec['unit']
        assert failure['foreign_processes_signalled'] is False and failure['upstream_retriever_signalled'] is False
        assert not (job / 'complete.json').exists() and not list(job.rglob('model-initialized.json'))
        progress = read(job / 'progress.json') if (job / 'progress.json').exists() else {}
        assert all(progress.get(key, 0) == 0 for key in ('completed_own_batches', 'questions', 'trajectories'))
        child_path = job / 'children.json';records[child_path] = original(view, child_path);children = read(child_path)
        child_pids = children['own_process_groups'];assert all(type(pid) is int and pid > 0 for pid in child_pids)
        pids = [spec['pid'], *child_pids];assert len(pids) == len(set(pids))
        borrowed = {allocation['source_owner']['pid'], allocation['local_relay_process']['pid']}
        assert not borrowed.intersection(pids), 'Borrowed healthy owner/relay must never be an own failed PID'
        evidence = read(view.path(allocation['allocation_evidence']))
        assert evidence['node'] == node and evidence['unit'] == spec['unit'] and evidence['main_pid'] == spec['pid']
        assert evidence['cgroup'] == allocation['native_cgroup']
        for pid in pids:
            assert pid not in known
            birth = evidence.get('controller_birth') if pid == spec['pid'] else None
            assert birth is None or type(birth) is int and birth > 0
            known[pid] = dict(pid=pid, birth=str(birth) if birth is not None else None,
                source=view.canonical(view.path(allocation['allocation_evidence'])) if birth is not None else view.canonical(job / ('allocation.json' if pid == spec['pid'] else 'children.json')),
                known_birth_not_stored=birth is None)
        facts = []
        for relative, _ in spec['pins']:
            if '/sampler-launches/' in relative and relative.endswith('/failure.json'):
                fact = read(view.path(BASE / relative))
                assert (fact['error_type'] == 'OutOfMemoryError' and 'CUDA out of memory' in fact['error']) or (
                    fact['error_type'] == 'ValueError' and fact['error'] == 'Observed free VRAM cannot accommodate the configured model reserve')
                facts.append(dict(path=str(BASE / relative), error_type=fact['error_type'], error=fact['error']))
        assert facts
        native = view.path(BASE / spec['native']);stage = view.path(BASE / spec['stage']);roots.extend((job, native, stage))
        paths.update(metadata((job, native, stage)))
        control = stage.parent
        for name in ('submit-intent.json', 'submit-receipt.json', 'submit-context.json', 'stage-receipt.json', 'slot-context.json'):
            path = control / name
            if path.is_file():paths.add(path)
        rows.append(dict(key=spec['key'], node=node, unit=spec['unit'], controller_pid=spec['pid'],
            job_directory=str(BASE / spec['job']), allocation_path=str(BASE / spec['job'] / 'allocation.json'),
            allocation_sha256=spec['allocation_sha256'], failure_path=str(BASE / spec['job'] / 'failure.json'),
            failure_sha256=spec['failure_sha256'], children_path=view.canonical(child_path), children_sha256=records[child_path]['sha256'],
            children_pinned_before_seal=False, child_pids=child_pids, capacity_failure_facts=facts,
            original_snapshot=spec['snapshot'], actual_model_initialized=False, new_trajectories=0))
    return rows, records, roots, paths, known


def owned_cgroup_pids(unit):
    group = '/system.slice/' + unit
    found = set()
    for process in Path('/proc').glob('[0-9]*'):
        try:
            paths = [line.split(':', 2)[-1] for line in (process / 'cgroup').read_text().splitlines()]
            if any(path == group or path.startswith(group + '/') for path in paths):found.add(int(process.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):pass
    return found


def release(rows, known):
    units = []
    for row in rows:
        unit = row['unit']
        raw = subprocess.check_output(['systemctl', 'show', unit, '-p', 'Id', '-p', 'MainPID', '-p', 'ActiveState', '-p', 'ControlGroup', '-p', 'Result', '-p', 'ExecMainStatus'], text=True, timeout=15)
        props = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
        assert props['Id'] == unit and props['MainPID'] == '0' and props['ActiveState'] in ('failed', 'inactive')
        assert props['ControlGroup'] in ('', '/system.slice/' + unit)
        folder = Path('/sys/fs/cgroup/system.slice') / unit
        pids = {int(value) for path in folder.rglob('cgroup.procs') for value in path.read_text().split()} if folder.exists() else set()
        pids = sorted(pids | owned_cgroup_pids(unit))
        assert not pids
        units.append(dict(unit=unit, observed=props, remaining_cgroup_pids=pids))
    remaining = sorted(pid for pid in known if Path('/proc/%d' % pid).exists())
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=20)
    gpu = sorted(set(known) & {int(line.strip()) for line in apps.splitlines() if line.strip().isdigit()})
    assert not remaining and not gpu
    return dict(epoch=time.time(), units=units, remaining_known_pids=remaining, remaining_original_births={}, own_gpu_contexts=gpu,
        original_births_gone=True, own_cgroups_empty=True, known_birth_not_stored=any(row['birth'] is None for row in known.values()),
        birth_claim_basis='Every evidenced own numeric PID is absent; no missing historical birth tick was inferred', foreign_signalled=False)


def publish(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode();temp = path.with_name(path.name + '.tmp-%d-%d' % (os.getpid(), time.time_ns()))
    try:
        with temp.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temp, path)
        except FileExistsError:assert path.read_bytes() == raw
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temp.unlink(missing_ok=True)


def summary(path, value):
    assert value['status'] == STATUS and value['run_id'] == RUN and value['node'] in ('h100-5', 'h100-10')
    return dict(status=STATUS, node=value['node'], proof=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        units=len(value['units']), original_files=len(value['original_files']), original_bytes=value['original_bytes'],
        known_owned_pids=len(value['pid_evidence']), historical_births_missing=value['pids_without_recorded_historical_birth'],
        controlled_stop=False, healthy_workloads_modified=False, global_goal_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('action', choices=('plan', 'seal'), nargs='?', default='plan')
    parser.add_argument('--node', choices=('h100-5', 'h100-10'), required=True);parser.add_argument('--base', default=str(BASE));parser.add_argument('--execute', action='store_true')
    args = parser.parse_args();rows = selected(args.node);view = View(args.base)
    relative = 'preparation/owner-v2-spare-admission-terminal-%s-v1' % args.node
    root = view.base / relative;complete = root / 'complete.json'
    if args.action == 'plan':
        print(json.dumps(dict(status='LOCAL_STATIC_PLAN_ONLY', node=args.node, units=rows, proof=str(BASE / relative / 'complete.json'), children_pinned_before_seal=False, controlled_stop=False)));return
    assert args.execute and os.geteuid() == 0 and socket.gethostname().split('.')[0] == args.node
    assert read(view.base / 'config.json')['run_id'] == RUN
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'seal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete.exists():print(json.dumps(dict(summary(complete, read(complete)), existing_proof_read_no_rehash=True)));return
        rows, records, roots, paths, known = prepare(view, args.node)
        observations = [release(rows, known)]
        for path in sorted(paths):
            if path not in records:records[path] = original(view, path)
        originals = sorted(records.values(), key=lambda row: row['path']);stable(view, originals)
        time.sleep(3);observations.append(release(rows, known));assert observations[1]['epoch'] - observations[0]['epoch'] >= 3
        assert metadata(roots).issubset(paths);stable(view, originals)
        value = dict(status=STATUS, run_id=RUN, node=args.node, units=rows, termination_kind='ACTUAL_CAPACITY_FAILURE_NOT_CONTROLLED_STOP',
            original_files=originals, original_bytes=sum(row['bytes'] for row in originals), release_observations=observations,
            pid_evidence=sorted(known.values(), key=lambda row: row['pid']),
            original_owned_births={str(pid): row['birth'] for pid, row in known.items() if row['birth'] is not None},
            pids_without_recorded_historical_birth=sorted(pid for pid, row in known.items() if row['birth'] is None),
            known_birth_not_stored=any(row['birth'] is None for row in known.values()),
            child_pid_scope='Retained own_process_groups only; earlier reaped launch PIDs may not have been retained',
            original_failures_not_rewritten=True, original_shared_partials_retained=True, actual_model_initialized=False,
            new_trajectories=0, controlled_stop=False, foreign_signalled=False, healthy_workloads_modified=False,
            source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), global_goal_complete=False, epoch=time.time())
        publish(complete, value);print(json.dumps(summary(complete, value)))


if __name__ == '__main__':main()
