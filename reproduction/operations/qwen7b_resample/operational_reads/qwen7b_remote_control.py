import importlib.util,os,subprocess,sys
for n,p in [('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py'),('route','/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018/reproduction/operations/h100_reference_replica/readonly_management_route.py')]:
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);globals()[n]=m
os.environ['PREFIX5_AB_MGMT_EXISTING_PROXY']='1'
r=subprocess.run(route.management_argv(ctl.route(sys.argv[1],'/usr/bin/python3 -')),input=open(sys.argv[2]).read(),text=True,capture_output=True,timeout=110)
print(r.stdout);print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
