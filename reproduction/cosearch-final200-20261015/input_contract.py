"""Bind unchanged shared prompts to the official final references in file order."""
import json
from common import TRAIN, N
from common_em import identity
from question_metadata import with_question


def load_inputs():
    import pandas as pd
    frame=pd.read_parquet(TRAIN/'data/final.parquet')
    references=[json.loads(line) for line in (TRAIN/'data/final.jsonl').open() if line.strip()]
    assert len(frame)==len(references)==N
    assert len({x['question_id'] for x in references})==N
    metadata=[]
    for index,(_,row) in enumerate(frame.iterrows()):
        reference=references[index]
        restored=with_question(dict(raw_prompt=list(row['prompt']),
                                    extra_info=dict(row['extra_info'])))['extra_info']
        assert restored['question'].strip()==reference['question'].strip(),index
        assert identity(reference['question'],row['reward_model']['ground_truth'])==identity(
            reference['question'],reference['answers']),index
        assert str(row['data_source'])==reference['data_source'],index
        metadata.append(restored)
    return frame,references,metadata
