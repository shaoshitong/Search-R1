"""One local integration contract: execute the prepare loop from its AST."""
import ast,copy,unittest
from pathlib import Path
from .test_peer_long_prefix_diagnostic import fixture
from .peer_long_prefix_diagnostic import reconstruct

class FullJournalSchema(unittest.TestCase):
 def test_actual_prepare_policy_loop_matches_existing_reconstruct20(self):
  source=Path(__file__).with_name('online_packed_variance.py');tree=ast.parse(source.read_text());prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='prepare')
  # The exact runtime extraction statements, rather than a second implementation.
  loop=next(n for n in ast.walk(prepare)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='turn'and ast.unparse(n.iter)=="r['behavior_turns']")
  init=ast.Assign(targets=[ast.Name(id='policy',ctx=ast.Store())],value=ast.List(elts=[],ctx=ast.Load()))
  result=ast.Return(value=ast.Name(id='policy',ctx=ast.Load()));fn=ast.FunctionDef(name='actual_extract',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='r')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[init,copy.deepcopy(loop),result],decorator_list=[])
  module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope={};exec(compile(module,'exact_prepare_loop','exec'),scope)
  for first,second in((25,90),(3,90),(3,4)):
   full=fixture(first=first,second=second,observation=(800,801));expected=reconstruct(full,20)['tokens'];actual=scope['actual_extract'](full);self.assertEqual(actual,expected);self.assertNotIn(800,actual);self.assertNotIn(801,actual);self.assertEqual(len(actual),min(20,first+second))
if __name__=='__main__':unittest.main()
