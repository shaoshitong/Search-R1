"""Small synthetic CPU-only loss/gradient timing; no real tensors or remote IO."""
import json,platform,resource,time
from pathlib import Path
import torch
from reproduction.operations.qwen7b_warmup.exact_uniform_g5_loss import exact_uniform_g5_loss

def main():
    torch.set_num_threads(1);torch.manual_seed(1729);rows=[]
    for size in (10,20):
        groups=[dict(qid='synthetic-q%d'%q,trajectory_ids=['synthetic-q%d:t%d'%(q,i) for i in range(size)],labels=[int(i<q%(size+1)) for i in range(size)]) for q in range(48)];z=torch.randn(48,size,requires_grad=True);before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;start=time.perf_counter();result=exact_uniform_g5_loss(z,groups,combination_chunk=64);gradient=torch.autograd.grad(result['loss'],z)[0];after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rows.append(dict(source_G=size,questions=48,chunk=64,CPU_seconds=time.perf_counter()-start,combinations_per_question=result['combinations_per_question'],maximum_resident_G5_rows=result['maximum_resident_G5_rows'],saved_VJP_elements=result['retained_loss_gradient_elements'],peak_process_RSS_native_units=after,increment_peak_RSS_native_units=after-before,ru_maxrss_unit='bytes' if platform.system()=='Darwin' else 'KiB',finite_gradient=bool(torch.isfinite(gradient).all()),native_model_forwards=0))
    print(json.dumps(dict(status='SYNTHETIC_CPU1_NOT_GPU_QUALIFICATION',platform=platform.platform(),torch=torch.__version__,threads=1,real_data_read=False,remote_actions=False,results=rows)))
if __name__=='__main__':main()
