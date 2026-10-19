"""Routing/once-scope tests; no SSH, actual data, model or GPU use."""
import unittest
from .assigned_cpu_control import namespace_command,outer_script,execute


class AssignedCPUControlTests(unittest.TestCase):
    def entry(self,node,arg=False):
        return dict(node=node,runtime_wrapper_path='/root/.local/share/search-g1/source8-fault-recovery/qualified/paths.sh',
                    runtime_wrapper_node_argument=arg,required_process_environment={'SEARCH_G1_B_PREFIX_TOKENS':'80'if node=='h100-5'else'5'})

    def test_distinct_actual_nodes_keep_their_qualified_wrapper_calling_conventions(self):
        a=namespace_command(self.entry('h100-11',True),'A worker')
        b=namespace_command(self.entry('h100-5'),'B worker')
        self.assertEqual(a[6:],[ 'h100-11','/usr/bin/python3','-c','A worker'])
        self.assertEqual(b[6:],[ '/usr/bin/python3','-c','B worker'])
        for node in ('h100-11','h100-5'):
            code=outer_script(self.entry(node,node=='h100-11'),'print("read only")')
            compile(code,'cpu_control','exec')
            self.assertIn("'CUDA_VISIBLE_DEVICES': ''",code)
            self.assertIn("'SEARCH_G1_B_PREFIX_TOKENS': '80'"if node=='h100-5'else"'SEARCH_G1_B_PREFIX_TOKENS': '5'",code)

    def test_mutating_retry_and_wrong_node_route_are_rejected_before_any_ssh(self):
        arms={'A':self.entry('h100-11',True),'B':self.entry('h100-5')}
        with self.assertRaises(ValueError):execute(arms,lambda _: 'unused','start','retained18_fallback_lan5','test')
        with self.assertRaises(ValueError):execute(arms,lambda _: 'unused','start','lan5','test')
        with self.assertRaises(ValueError):namespace_command(dict(self.entry('h100-5'),node='h100-8'),'unused')


if __name__=='__main__':unittest.main()
