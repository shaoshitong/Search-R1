"""Read a committed B periodic pair and next EMA adoption; CPU metadata only."""
import argparse
import json
from pathlib import Path

from .assigned_cpu_control import execute
from .follow_h10011_ab_normal_boundary import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int, required=True)
    args = parser.parse_args()
    if args.outer not in range(60, 200, 10):
        parser.error('Require an actually committed progressive periodic60..190')
    entry = json.loads((ROOT/'work/prefix5-ab-assigned-arms.json').read_text())['B']
    reader = Path(__file__).with_name('once_records_20261018')/'read_progressive_period_pair.py.txt'
    source = reader.read_text()

    def inner(arms):
        return ('entry='+repr(entry)+'\nsource='+repr(source)+'\nouter='+repr(args.outer)+'\n'+
            "import json,hashlib\nfrom pathlib import Path\n"
            "p=Path(entry['base'])/'preparation/launch-config.json'\n"
            "assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['config_sha256']\n"
            "namespace={}\nexec(compile(source,'existing_periodic_metadata_reader','exec'),namespace)\n"
            "result=namespace['observe']({'base':entry['base'],'outer':outer})\n"
            "print(json.dumps({'B':result}))\n")

    value = execute({'B':entry}, inner, 'status', 'retained18',
        'period%d-original-pair-metadata'%args.outer)
    output = ROOT/'work'/('prefix5-ab-period%d-original-pair-small.json'%args.outer)
    output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), evidence=value['evidence'],
        outer=args.outer, role=value['arms']['B']['role'], training_modified=False)))


if __name__ == '__main__':
    main()
