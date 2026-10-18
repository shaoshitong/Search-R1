"""Set the immutable phase dimensions before importing native model code."""
import importlib
import os
import sys

from .schedule import outer_plan


def main():
    role=sys.argv[1];args=sys.argv[2:]
    outer=int(args[args.index('--outer')+1]);plan=outer_plan(outer)
    os.environ['SEARCH_G1_GRPO_FROM50']='1'
    os.environ['SEARCH_G1_GRPO_PREFIX']=str(plan['prefix_tokens'])
    modules=dict(rollout='reproduction.prefix_selection200.rollout',
        probe='reproduction.prefix_selection200.probe',
        periodic='reproduction.prefix_selection200.periodic_data',
        actor='reproduction.actor50_grpo.actor',lift40='reproduction.actor50_grpo.lift40')
    if role=='rerank':
        from .rerank import main as execute
    elif role=='periodic' and args[0]=='complete-pool' and plan['actual_em_rerank']:
        from .rerank import complete_main as execute
    else:execute=importlib.import_module(modules[role]).main
    sys.argv=[modules.get(role,role)]+args
    execute()


if __name__=='__main__':main()
