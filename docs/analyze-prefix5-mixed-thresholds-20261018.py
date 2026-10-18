"""Reproduce p>=0.5 versus q>=0.5 confusion from remote-only saved scalar rows.

Diagnostic thresholds do not describe the runtime's top512 ranking decision.
"""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time


def main():
    root=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2/analysis/probe-mixed-diagnostics-20261018-v1')
    data=root/'periodic-all-candidate-records.csv.gz'
    results=[]
    for name,outers in [('early10',[10]),('late100',[100]),('late110',[110]),
                        ('late100_110',[100,110]),('all10_110',list(range(10,111,10)))]:
        matrix=collections.Counter();qmatrix=collections.Counter();equal=0
        with gzip.open(data,'rt') as handle:
            for row in csv.DictReader(handle):
                if int(row['outer']) not in outers:continue
                k=int(row['K']);truth='all_wrong' if k==0 else 'all_correct' if k==5 else 'mixed'
                ps=[float(row['p'+str(j)]) for j in range(5)]
                count=sum(p>=.5 for p in ps)
                prediction='all_wrong' if count==0 else 'all_correct' if count==5 else 'mixed'
                matrix[(truth,prediction)]+=1
                qmatrix[(truth,'mixed' if float(row['q'])>=.5 else 'nonmixed')]+=1
                equal+=truth=='mixed' and max(ps)-min(ps)<1e-10
        results.append(dict(scope=name,outers=outers,
            p_binary_group_confusion={t:{p:matrix[(t,p)] for p in ['all_wrong','mixed','all_correct']}
                                      for t in ['all_wrong','mixed','all_correct']},
            q_binary_mixed_confusion={t:{p:qmatrix[(t,p)] for p in ['nonmixed','mixed']}
                                      for t in ['all_wrong','mixed','all_correct']},
            true_mixed_identical_five_p=equal))
    result=dict(epoch=time.time(),p_threshold=.5,q_threshold=.5,rows=results,
                source_csv_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                runtime_decision='q top512 ranking, not p threshold classification',training_modified=False)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
