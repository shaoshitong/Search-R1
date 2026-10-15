"""CPU readback of the actual full-size inherited probe before GPU admission.

Only a small receipt is written. This is not a CUDA or student update test.
"""
import json
import os
from pathlib import Path
import resource
import time

from .common import BASE, read, publish


def main():
    import torch
    from search_g1.full_state_restore import equal_tree
    from search_g1.qprobe_periodic_model import PeriodicQwenProbe,restore_engine,core_config
    from search_g1.qprobe_periodic_features import QuestionMetadata
    from reproduction.hpc3_dynrpo_periodic.bind import recipe
    from reproduction.hpc3_dynrpo_periodic.warmup import PREVIOUS
    torch.set_num_threads(1)
    started=time.monotonic()
    saved=torch.load(BASE/'warmup/continued-head.pt',map_location='cpu',weights_only=True)
    engine=restore_engine(saved['head'],epochs=20)
    head=PeriodicQwenProbe(engine,saved['token_feature_spec_sha256'],completed_outer=0,
        actor_snapshot_id='cpu-production-readback:0',phases=[dict(outer=0,fit_epochs=20,
        head_adam_steps_before=0,head_adam_steps_after=400,inherited_first_ten_epochs=True)])
    head.validate_state()
    if head.config!=core_config(recipe()):
        raise ValueError('Actual saved probe differs from the effective formal configuration')
    equal_tree(engine.model.state_dict(),saved['head']['model'])
    equal_tree(engine.optimizer.state_dict(),saved['head']['optimizer'])
    if not torch.equal(engine.rng.get_state(),saved['head']['private_rng']):
        raise ValueError('Actual full-size probe private RNG was not inherited')
    metadata=QuestionMetadata(PREVIOUS/'token-features')
    ready=read(BASE/'warmup/ready.json')
    if metadata.spec_sha!=head.feature_spec_sha or ready['head_adam_steps']!=head._adam_steps():
        raise ValueError('Actual probe and accepted question representation differ')
    if torch.cuda.is_initialized():
        raise ValueError('CPU readback must not initialize an unallocated GPU')
    value=dict(stage='ACTUAL_PRODUCTION_PROBE_CPU_READBACK_PASS',
        deployed_source_commit=read(BASE/'preparation/source-publication.json')['commit'],
        head_sha256=ready['head_sha256'],head_version=head.version,
        head_adam_steps=head._adam_steps(),fit_epochs=head.config.epochs,
        trainable_tensors=len(list(head.model.parameters())),
        trainable_parameters=sum(p.numel() for p in head.model.parameters() if p.requires_grad),
        trainable_modules=['existing_native_last_decoder_block','existing_readout'],
        actual_model_equal=True,actual_adam_equal=True,actual_private_rng_equal=True,
        student_parameters_copied=False,optimizer_reset=False,canonical_question_count=len(metadata.index),
        initial_feature_tensor_bank_opened=False,cuda_initialized=False,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        elapsed_seconds=time.monotonic()-started,checked_unix=time.time())
    publish(BASE/'preparation/production-probe-cpu-readback.json',value)
    print(json.dumps(value))


if __name__=='__main__':main()
