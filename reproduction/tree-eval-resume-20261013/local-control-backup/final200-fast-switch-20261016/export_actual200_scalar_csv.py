"""Export all 30 recorded series without altering preserved historical curves."""
import csv
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / 'outputs/tree-nested-branch-handoff-20261013'
source = BASE / 'METRIC_RECORD_COVERAGE_THROUGH200_ORIGINAL_RECORDS.json'
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '37b80e80f7318faa66d81ac316d4078405fc895c1ea3038ad51c9a9b808944e3'
data = json.loads(raw)
series = {}
for name, value in data['coverage'].items():
    assert not value['unknown_outers']
    series[name] = value['values_in_outer1_to200_order']
series.update(data['original_denominator_and_count_series'])
series.update(data['additional_original_recorded_fields_1_to200'])
assert len(series) == 30
assert all(len(values) == 200 and all(type(v) in (int, float) and math.isfinite(v)
                                    for v in values) for values in series.values())
output = BASE / 'ACTUAL200_COMPLETE_SCALAR_SERIES.csv'
assert not output.exists()
with output.open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['outer'] + list(series), lineterminator='\n')
    writer.writeheader()
    for i in range(200):
        writer.writerow(dict(outer=i+1, **{k: values[i] for k, values in series.items()}))
with output.open(newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 200
for i, row in enumerate(rows):
    assert int(row['outer']) == i+1
    assert all(float(row[k]) == values[i] for k, values in series.items())
receipt = dict(schema='tree_actual200_complete_scalar_csv_v1',
    source=dict(path=str(source), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
    output=dict(path=str(output), bytes=output.stat().st_size,
                sha256=hashlib.sha256(output.read_bytes()).hexdigest()),
    rows=200, recorded_scalar_columns=30, roundtrip_values_checked=6000,
    missing_values=0, imputation=False, historical_curve_modified=False,
    scope='Original recorded scalar series; preserves their definitions and source bindings in the input JSON.')
path = BASE / 'ACTUAL200_COMPLETE_SCALAR_SERIES_RECEIPT.json'
assert not path.exists()
path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
