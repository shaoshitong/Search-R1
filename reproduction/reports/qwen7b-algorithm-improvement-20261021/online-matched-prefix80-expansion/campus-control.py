import pathlib,shlex,subprocess,sys
script=pathlib.Path(sys.argv[1]);r=subprocess.run(['python3',str(next(p/'work/hpc3_private_control.py' for p in pathlib.Path(__file__).resolve().parents if(p/'work/hpc3_private_control.py').is_file()))],input='python3 -c '+shlex.quote(script.read_text()),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
sys.stdout.write(r.stdout);sys.stderr.write(r.stderr);raise SystemExit(r.returncode)
