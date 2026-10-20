"""Read-only remote CPU reconstruction of every completed fixed-G5 feature row."""
import importlib.util,json,pathlib,shlex,subprocess
CONTROL=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
ROOT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/validation-fixedG5-hf-replay-20261021'
CODE=r'''
import hashlib,json,pathlib,time,torch
from reproduction.operations.qwen7b_warmup import validation_fixedg5_replay as m
from reproduction.operations.qwen7b_warmup.replay_feature_reader import ReplayFeatureReader
torch.set_num_threads(2);torch.set_num_interop_threads(1)
b=m.TARGET;cp=b/'capture/complete.json';before_cp=m.ref(cp);done=m.read(cp);started=time.time()
assert done['status']==m.STATUS and done['questions']==2048 and done['trajectories']==10240 and done['capture_layers']==[26]
for key in ('source_selection','source_complete','source_fixed_G5','features'):m.checked(done[key])
assert done['source_complete']['sha256']==m.SOURCE_SHA
plan=m.read(done['source_fixed_G5']['path']);declarations=m.fixed_declarations(plan);ids=[x['trajectory_id'] for x in declarations]
assert done['fixed_G5_sha256']==plan['summary']['selected_g5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'
selection=m.read(done['source_selection']['path']);assert selection['trajectories']==declarations
manifests=m.read(done['features']['path']);assert manifests==done['feature_manifests_by_capture_layer']['26'] and len(manifests)==len(done['chunks'])==32
reader=ReplayFeatureReader(manifests,26);assert set(reader.records)==set(ids)
for start in range(0,len(ids),8):
 batch=reader.get(ids[start:start+8],'cpu');assert batch['trajectory_ids']==ids[start:start+8]
 assert batch['features'].device.type=='cpu';del batch
 if (start+8)%1024==0:m.r.atomic(b/'reader-audit-progress.json',dict(rows=start+8,expected_rows=10240,time=time.time()))
reader.close()
assert m.ref(cp)==before_cp
value=dict(status='ALL_10240_NEW_FIXED_G5_HF_ROWS_RECONSTRUCTED_RETAINED_SHA_PASSED',complete=m.ref(cp),features=done['features'],source_complete=done['source_complete'],source_groups=done['source_groups'],source_fixed_G5=done['source_fixed_G5'],fixed_G5_sha256=done['fixed_G5_sha256'],questions=2048,trajectories=len(ids),manifests=len(manifests),chunks=len(done['chunks']),capture_layer=26,qualified_batching=done['batching'],all_manifest_and_payload_SHA256=True,all_reconstructed_row_SHA256=True,ordered_qid_sha256=m.digest([g['qid'] for g in plan['groups']]),ordered_trajectory_sha256=m.digest(ids),labels_read=False,final_pool_read=False,model_tensors_loaded=False,actor_updates=0,probe_updates=0,elapsed_seconds=time.time()-started,time=time.time())
m.once(b/'independent-reader-audit.json',value);print(json.dumps(value))
'''
spec=importlib.util.spec_from_file_location('new_fixed_g5_full_reader_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
code_root=ROOT+'/code-70f43e7708ea';command='env QWEN7B_CODE_ROOT='+shlex.quote(code_root)+" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 bash "+shlex.quote(code_root+'/reproduction/operations/qwen7b_warmup/runtime.sh')+' -c '+shlex.quote(CODE)
assert len(command.encode())<16000
ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();result=m.run(command,account='mou176',source_address=ip,timeout=300)
if result.returncode:raise RuntimeError('Remote full-row reconstruction audit failed: '+result.stderr.decode(errors='replace')[-1000:]+' '+result.stdout.decode(errors='replace')[-1000:])
value=json.loads(result.stdout);path=pathlib.Path(__file__).with_name('independent-all-rows-audit.json');path.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
