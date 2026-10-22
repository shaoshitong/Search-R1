"""Small pure-CPU admission checks; no Torch, remote input or inference."""
import hashlib,json,time
from . import online_packed_full169028_MSE_P80 as m
from . import deploy_full169028_MSE_P80 as d

def main():
 ref=dict(path='/remote/synthetic-contract.json',bytes=1,sha256='b'*64)
 for item in({},dict(actual_physical_execution_and_seal_source_exact=False)):
  try:m.full_acceptance_physical_source_ref(item)
  except ValueError:pass
  else:raise AssertionError('Unresolved physical source accepted')
 assert m.full_acceptance_physical_source_ref(dict(actual_physical_execution_and_seal_source_exact=True,physical_source_contract=ref))==ref
 valid=dict(status='ALL169028_OWN_G10_PHYSICAL_SEAL_EXECUTION_AND_OWN_EM_RESOLVED',all_selected_own_EM_matches_canonical=True,whole_G5_not_mixed=True,selection_used_EM=False);m.validate_physical_source_contract(valid)
 for key in valid:
  wrong=dict(valid);wrong.pop(key)
  try:m.validate_physical_source_contract(wrong)
  except ValueError:pass
  else:raise AssertionError(key)
 for item in({},dict(fresh_full_P20_prefix_prompt_and_own_EM_exact=False)):
  try:m.full_acceptance_alignment_ref(item)
  except ValueError:pass
  else:raise AssertionError('P20 exact gate was weakened')
 assert d.FIXTURE.endswith('/full169028-policy80-physical-ownG10-20261023-v2/fixture')and d.BASE.endswith('-20261023-v3')
 print(json.dumps(dict(status='FULL_PHYSICAL_SOURCE_AND_P20_GATES_TARGETED_CPU_PASS',time=time.time(),actual_physical_execution_flag_required=True,exact_physical_source_contract_field_required=True,four_contract_fields_required=list(valid),P20_exact_gate_preserved=True,new_fixture=d.FIXTURE,new_base_seed1=d.BASE,new_base_seed2=d.BASE_SECOND,training_math_or_geometry_changed=False,runtime_SHA256=hashlib.sha256(open(m.__file__,'rb').read()).hexdigest(),no_GPU_or_Slurm_or_real_assets=True,final4096_read=False)))
if __name__=='__main__':main()
