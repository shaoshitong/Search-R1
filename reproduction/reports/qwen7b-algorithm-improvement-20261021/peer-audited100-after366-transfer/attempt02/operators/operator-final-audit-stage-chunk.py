v={'root': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-delta-import-runtime-audited100-after366-20261021-636015f5-a02/independent-audit-e21e61e8fbfa', 'refs': {'audit.py': {'bytes': 8820, 'sha256': 'f5b43eb257050642d1f15548d25f5ea14a29a7fb6d1cfedc3b93bd9a06b0a1e2'}, 'selection.json': {'bytes': 170253, 'sha256': '00e5021284cb8f6614670452b749c7add10c028e50578822432600762074662a'}}, 'commit': 'e21e61e8fbfac1674b6e7c04bbb5218107577bd5', 'ingest': '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed-ingest-audited100-after366-20261021-636015f5-a02', 'selection_sha256': '00e5021284cb8f6614670452b749c7add10c028e50578822432600762074662a', 'record_sha256': '636015f5b3cf84ad49688a1cee6b8d15959ccbc59451e91f6dae5d625df103ce'}

import base64,gzip,hashlib,json,pathlib,os,subprocess,sys,time
b=pathlib.Path(v['root']);b.mkdir(parents=True,exist_ok=True)
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def atomic(p,val):
 q=p.with_name(p.name+'.tmp-'+str(os.getpid()));q.write_text(json.dumps(val,sort_keys=True));q.replace(p)
part={'offset': 60000, 'data': 'LsLN3/Qc+G708/Tf4Azejwfmh10/94Off013gLnFNHqvmFeG+r7ffB2fU7xDv8bkWB/4Ntrou5v/86efq4C7+zY0nOP5pzxoyWNvw9Xow8+L44dzYDFY5/tXi8P98/x1F07PH95vjp/j7kR5ePpLSaxXHvjwLviHPvv5Q1S2H3J912++6Bnf/aDefHN+m4/aXWT95a/kA+7s2/zTedBnm/+qc/LLbjb59ctnl9RbxblKte3F2eWX+v2+RG+7m4DHHKX5Cp0F/IOGNmOOFdfU76h3P5s+djY/FR171JadsWrqYpDiIVbD3+cO452xtnj+h+/h07n63L/8AXepekWx64/zP8uFk8s9+RfPajf7j19/+nFcu/NXz7TvSxQD0NN8m690/vU9Z1/kSeI1++3wsu7jl/gRfCELdDTxQmPmj/dZvlE/qzyHDmT9dFS8Ck271cO/GKt/xouV6kl/N0fHVfV1hn7Klp3js8+bclW4Of72XlX//HP3flErZim/Tz/+7plF4Fjcn+DK7C6+u1wz6P5EViu8CQY6x3/fzpz/6nmELztrOmfVhwuazOw14Eup1joyB+rx6E8nnjf40mRm2FM9B57RHO+vAI0IPKsSZhum/Q7Peo/+aIQO+bNuiGJQ0m5/XBnf1rU/KM4tdFdWf/vi7P9uTrQ9/85v8H53Phq3es0vn/Xfif3/ynO+fqcL1Ru7Fg9A1VZRAaY3K2y/A0exsJlGgf5RwH1P2Ff4rc7R8uiy5sjvJR5OjfKux1Nqeijj5zv9KQcxry1CPf/up/fq0cDCTX/4HKMvP3vd76fTn83o/vXz+jkGfPns0KpV3xK9RXsjRwu6Od1yv/EaQw/O5kxgXT17GnZmxp89Ma+nzyhtFua40+i9Ko5uaud3uu9N0GvtFoGLV+A5wv5+f3qfwVcY1ee6xfWeOcc/zo6WYRovbX77VSz9s1k62jp/rTnoFa1ufUfofGR7YqXt5plZFoY5N50mPCOIp6PCP0VlPT6wn7f9aLgeUvvDr0V7xcHJ7CZRGvdah5+f1XOf9rlXpt7zunO8l99i7G/z5d/P7YhXz7O759eLFoFqx3+wC5nXW92d/dD0ibNxBy4p4X0wF0XjAW94dNTiflf6u9f457btc/77iBmU4lmU/O1D/kPTY0+/mm18umP9XbC5xjn9/R6oXfxX3mzS90HyX5vNIl5tnjGyk82E+PN+MJ3pv9/p3/FysFnar+s+vx8E081kOlluVuP3gx/SbNp86wzm099RbOuUG1Ubj3i+VRnqrKkefR9+kef0s5XXBqqz+7+bUzvbfuCL+jj4lMMvY8eVc38c/Oe//c//AoijJIbSpAMA', 'sha': 'bca2e90f6a01bc9bdfd999b42add5f3d783dc04747d0334a6e27cab821d1ca6d'}

p=b/'stage.gz.upload';data=base64.b64decode(part['data']);assert hashlib.sha256(data).hexdigest()==part['sha'];size=p.stat().st_size if p.exists() else 0
if size==part['offset']:
 with p.open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
else:
 assert size>=part['offset']+len(data)
 with p.open('rb') as f:f.seek(part['offset']);assert f.read(len(data))==data
print(json.dumps(dict(status='PUBLISHED_AUDIT_METADATA_CHUNK_ACCEPTED',offset=part['offset'])))
