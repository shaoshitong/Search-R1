"""Bootstrap CLI which also works when the old reproduction package is regular."""
import argparse
import json
from pathlib import Path
import sys
from legacy_eval_control import activate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('pipeline','runner'))
    args,rest=parser.parse_known_args()
    locate=argparse.ArgumentParser(add_help=False)
    if args.command=='pipeline':
        locate.add_argument('--base',type=Path,required=True);known,_=locate.parse_known_args(rest)
        document=json.loads((known.base/'preparation/legacy-evaluation-manifest.json').read_text())
    else:
        locate.add_argument('--binding',type=Path,required=True);known,_=locate.parse_known_args(rest)
        document=json.loads(known.binding.read_text())
    activate(document['legacy_source_root'])
    import importlib
    module=importlib.import_module('reproduction.legacy_evaluation.'+args.command)
    sys.argv=[sys.argv[0]]+rest
    return module.main()

if __name__=='__main__':main()
