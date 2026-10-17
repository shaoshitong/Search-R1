"""Import complete study epochs on the server; never import an unsaved partial."""
import json
import os
from pathlib import Path
import shutil
from reproduction.prefix20_online.launch import once, read, sha


def import_history(base,config):
    base=Path(base);spec=config['probe20k_study'];source=Path(spec['resume_fit_base'])
    receipt=base/'preparation/import-history-receipt.json'
    if receipt.exists():return read(receipt)
    proof_path=Path(spec['resume_fit_proof']);assert sha(proof_path)==spec['resume_fit_proof_sha256']
    proof=read(proof_path);assert proof['original_births_all_gone'] and proof['no_live_own_gpu']
    assert proof['base']==str(source) and proof['study_id']==spec['study_id']
    oldconfig=read(source/'preparation/launch-config.json')
    for key in ('actor_version','run_id'):assert oldconfig[key]==config[key]
    for key in ('dataset_base','dataset_seed','arms','epochs','learning_rate','ema_decay','source_probe_checkpoint'):
        assert oldconfig['probe20k_study'][key]==spec[key]
    imported=[]
    def link(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert dst.stat().st_size==src.stat().st_size;return
        try:os.link(src,dst)
        except OSError:shutil.copy2(src,dst)
    for arm in spec['arms']:
        name=arm['name'];record=proof['arms'][name]
        if not record:continue
        epoch=record['epoch'];old=source/'arms'/name;out=base/'arms'/name
        out.mkdir(parents=True,exist_ok=True)
        for e in range(epoch+1):
            for file in old.glob('epoch-%02d-*'%e):link(file,out/file.name)
            if e:link(old/('head-epoch-%02d.pt'%e),out/('head-epoch-%02d.pt'%e))
        for namefile in ('initialization.json','latest.json','latest-metrics.json'):
            src=old/namefile
            if src.exists():link(src,out/namefile)
        # latest-metrics may belong to an uncommitted epoch; explicitly select saved epoch.
        metrics=read(old/('epoch-%02d-metrics.json'%epoch))
        from reproduction.prefix20_online.launch import publish
        if (out/'latest-metrics.json').exists():(out/'latest-metrics.json').unlink()
        publish(out/'latest-metrics.json',metrics)
        for file in [*old.glob('rank-*-updates.jsonl'),old/'global-loss-updates.jsonl']:
            if not file.exists():continue
            rows=[json.loads(line) for line in file.open() if line.strip()]
            with (out/file.name).open('w')as f:
                for row in rows:
                    if row['epoch']<=epoch:f.write(json.dumps(row)+'\n')
        if epoch==spec['epochs']:link(old/'complete.json',out/'complete.json')
        imported.append(dict(arm=name,epoch=epoch,source_checkpoint=record['checkpoint'],
            actual_adam_per_rank=record['actual_adam_per_rank'],ema_updates=record['ema_updates'],unsaved_partial_imported=False))
    value=dict(source_base=str(source),proof=str(proof_path),arms=imported,server_side_only=True,actor_updated=False)
    once(receipt,value);return value
