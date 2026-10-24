import runpy,sys
root='/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-milestone100-150-native2-eval-20261025';commit='d5c8a152b709805ad36946759d60023191bad3d9'
sys.argv=['submit_milestone_eval.py','--root',root,'--commit',commit]
runpy.run_path(root+'/source/'+commit+'/reproduction/easy2560_joint4/submit_milestone_eval.py',run_name='__main__')
