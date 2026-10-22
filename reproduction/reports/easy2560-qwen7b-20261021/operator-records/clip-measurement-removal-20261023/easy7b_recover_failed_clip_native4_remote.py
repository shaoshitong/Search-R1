import io,json,os,subprocess,sys,time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022');commit='2d5183455551145c52d1783a38663e2fcf170ceb';old=base/'diagnostics/native4-clip-native4-paired-r1-20261023';new=base/'diagnostics/native4-clip-native4-paired-r2-20261023'
assert os.getuid()==14090 and (old/'job.id').read_text().strip()=='749812'
claim=json.loads((old/'submit-claim.json').read_text());assert claim['source_commit']==commit and claim['mode']=='clip-native4'
r=subprocess.run(['sacct','-n','-X','-j','749812','-o','JobIDRaw,User,State,ExitCode,NodeList','-P'],capture_output=True,text=True,timeout=20)
assert r.returncode==0 and r.stdout.strip()=='749812|jxu957|FAILED|1:0|ACD1-13',r.stdout+r.stderr
for rank in range(4):
 report=json.loads((old/('clip-comparison-rank-%d.json'%rank)).read_text());assert report['status']=='FAILED' and report['actual_clipping_calls']==0
sys.path.insert(0,str(base/'source'/commit))
from reproduction.easy2560_joint4.control import publish
from reproduction.operations.easy7b_20261021.submit_native4_fastpath_hpc import main
parent=new/'recovery-parent.json'
if not parent.exists():publish(parent,{'epoch':time.time(),'parent_job':'749812','parent_root':str(old),'actual_parent_accounting':r.stdout,'parent_actual_clipping_calls_all4':0,'reason':'Actual independent diagnostic CUDA initialization failure on ACD1-13; preserve old root and restore default node exclusion. Production749598 untouched.','same_runtime_commit':commit,'same_cp50_selection51_twenty_updates':True,'production_modified':False,'automatic_retry_loop':False})
request={'base':str(base),'source_base':'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022','commit':commit,'checkpoint':'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022/checkpoints/search-g1-common8/actor/global_step_50','modes':['clip-native4'],'suffix':'paired-r2-20261023'}
sys.stdin=io.StringIO(json.dumps(request));main()
