from pathlib import Path
import importlib.util,json,sys,datetime
D=Path(__file__).resolve().parent;target=sys.argv[1]
p=Path('/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016/reproduction/h1008_qprobe/management_transport.py');s=importlib.util.spec_from_file_location('transport',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
patterns= [('Prefix20 actor-hidden period20','search-g1-prefix-selection200-actorhidden20-20261016-a008d7ce9833*'),('Prefix20 frozen encoder every-outer','search-g1-prefix-selection200-20261016-5cba22ca409a*')] if target=='h100-8' else [('Prefix1024 diversity512','search-g1-prefix-diversity200-20261016-f3ee1435e501*')]
code='PATTERNS='+repr(patterns)+'\n'+(D/'read_prefix_scalar.py').read_text();tag=datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')
try:
 out=m.remote(code,D/(target+'-'+tag+'-transport.json'),target=target);(D/(target+'-current.json')).write_text(out);print(target,'saved',len(out))
except Exception as e:print(target,'read failed',str(e)[:1200])
