"""Immutable pure-CPU planning; no EM, tensors, fitting, or Slurm submission."""
import base64,fcntl,gzip,hashlib,json,pathlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
B=A/'peer-expanded-append-only-plan-v1'
COMMIT='efb7bb24a3e2d142fe1f92cb46f1ea363c39c993'
PAYLOAD='H4sIAAAAAAAC/5VZbW/bOBL+7l/B5HCQ1cp27GSzaVotkMvmrgV2kzZJ74thCLRE22xkSSGpvFzR/34zQ0qWZK+vlw+VJQ6H8z7PsIeHh59LJRgvCpElgzxLX1mR8mykDTeCxXlmFI+NDtizNKu8NCzNeSKzJTMi07nSLFcMaJXBb9/yuR72evcrYBjHojAiYWfjd5PBYym0kXk2mvxyOiiLBHkXSizkC5OayfW6NHyeCraS2uTqdchuUBCzEj2eJBJ38nQyPpucsIqTZmv+yuaCgXwyg3My8WLOT88m8OOZ2SN08O74jMXluky5kU9i2LvO2YI/CHY1ZmthOBDxgOWFkWv5H6GYElqYAB4pn4sUNApQPSXAIq8sX6BALE8TJ/qwd3h42JPrIleGrbhepXIefNN5FiieJfm61/v46e7+5vbT5cUf0ZevV3f3n26u70I0SO/+5r718fjoaHza+8fF/eXH8HjS3Pj18+8X91d3IViud3fz9fbyKrr4/eLPcHz06+SodzWu3k5OT3qXN39+vrm+ur6/C797ij97gSfWHP4FTdfwgE/RvFwshNJ2qfGmsqX3o9dLxIIlcglG7j/xtBT+eY/BnxKmVFml5VCvOMjTR2WHSbkutCUOICJM9CBedXiv8FUUXHHwqA77cETgnXt+AHEDERdxHUsZ/pOnWgQ8TfPnKOOZffeHIovzRPR9f7gSL04c3woXgZ8hVPvgh0AmIjPSvAZvAhs5MuZpVIdIYHLTep9zE6+cRuC7uzUczMRLkcpYGlB7DbJRbHFIiQVG9msGPjcyhnDXRr8Hz+dJGSM3VkgghNX1EMMAWS5VXhY6rKSaevaDN3tfKJmrcJeMoxEJRfvlgvV30fydSDAUOwptFnZt+y3skNMh1R9sSkXWtyL6B11aXAcTT736izc72KlBlyvtcgmIe0j3emGZ5vPmbqSotSAKLUSCX8e/Tt7t5P0kxbOOCqFqLkh+0lqPnoVcrgwuDCe/7GQjIPJjLAuRVd3uIL6g4lJmDTFJnuFONlQqdFRqkUQQMvUWYJII5c2wwGW5YRTZ9S4IvI0NojyLxYYSU8cFKaUel7Dx35hfV0rlqu9dvUBJhtq8LlKBJbZiNEA+TOelggdWcUjbx1IqsKdP7P7GrsUT1DmI3EIQC6XgmQmtmdUDBeRsDcmXMg3LUF9zFq/yXIuREqQSw6ppbYEpsgRuLvI3MkMkQyXtLyGuvnuPMoHUr+2CNQTeoW5/g7Oh2kcy0d6POiCn3aUZcBkfnbesv22UTy7rbA+DEjFQgqeDDS8nZNcmIJ0Op3AqijnrqjR7j+IKR9FSYZuW+BniZ7qL9G7ofVvBWZX+aAA0HArlg9r4Tr8r6+Cilahadm9NAuN2j4/eNFJ8X0R9cZqB3V4MxUBsSp6yhvVcVZMCLVgIbpz9QGqK6DhfYfF/JIUOQnxOz3fVi9n+0C6gd0JMV0WUJbmwaVFgd1ZPou7C9kSo3yBs+urEoVoC/iKZ3AsUYFHk8ar5NbJfPGv6BQCEcPpITpqjk9xO+vBIH2of9dGmbt1vlDeevdolv1HRKpfM/fpr9wx/V1lGgZyDcTu9ko/rt4OwCpS/MPMWX6vxQTg9mr0hqfeGxMeaZ4Wm2LrUWDOwgmAZeSUvVAdWfmBUzOzhhsvURsIuCc+tSQGvhSks95Ha7rP4aXhLj77MTN9hgKlXCKEGwkXJoAFbB7jdC7BrBPhz5gfjUwAQegUQJxV9OMZ3RQvRBoC7cArfpvJcviVP2HyW6Bk4fyn6RwGaH/c54DBrQqFExqbfCHrSM6jCb0oazf0tb8/e1ucHtfWbEWltYX/7b9FXFA/VJj+o7ddt2Rt+zmFdAoc1gm4PtoEZYOOlrrthtN1qw5Og0V+xuQb/TyOFLrrhvr91brBhp1NabFlzaYRWpTdFd7AZHeqFjilrHrZnRtgzo5hneUbsLMoNXexBsPtNmOkc2qGqknrDG8QA/dq6dTZRjGmRCkKWaBAdWeM4I+xS1g4hEYSjnYAcKWQMvEbQvAsRetdfIkyOSGSwaQX41niNsCuTpTAV6R9cLV13h1qWWKxZDUecUtxueE/lGFACqZqAHizmpQbiGojEeZlZhMKVhCHBc+DdYneycwvAt6eMnRB/VwUJdw1YQavuub9OJoSd+cvmbUjzlxMVBhqJQRO52ABV44cih2IUVaNjX/MnkQQAoYQsDAwhFWmeLeQSfRw0AwveN7PHrcDGWmEtC1po1nxW0oD2LiQDxhcGEZvR7HmVw4i8kPDP3ccLRgLBtI0M72kYH9lpbnR7/S+a3MFyGoZzCUOOAz1VX4dWahDcoZ+1fBmgp/IM6IkbQoqG8wAUZtR0cUyLOSJOBe6FaWnI7lc4v9sGbVbcMOgLcgFeQjNbkIi3AQtQhlXFAKdvqLAPdswClMpVvILyERAXIo1LAKYQQRTaeLNA9MPKeC5YLI7rjLr1NP8T8y4Mxu6aIrIwCr44j2HZwUMjuggBfFq1f5SxOvoD9WCKAv98u4VegOFTUGxzwzAibzsHOZNvIVLEGMRzCMnW98AW4FsPGv72rQAhgg2pNgLGTaBsXBR0UYCL1p/j3SL+Ge5NuessgAqB27ayY1uc/7llz3lVjrW3Vl87xBxBbdUWkPxoD+fafRFdEHn+1kzXIIapI8lLs5N0zxlzSK2HnZv2ojS6J4MUgrKEM2F963b5Gcwwukj4mu6JIAwzwZeCLs2wNMisrECaG3BIlulWQmxgb2P0OAg3N010+0CfD8KWM7uctvAoAmbUUmqYRw2Hzt5/CrRR9SzzBAednhCEekIIZRkN6apJ931/r2UuKlMsSkjDVpFzjLp5RzWmssNflIHWpEYbfHf18F1NPfzg5kLlcOSD/gEz8FEwDibB8Y9Kbao+5GKk6mzYq9Yt9B4gxIurdlG1Ag+QA8P64opqZfcmaq2yipJ/R+ZXXQvDp5nsLUzThVanZ5Ngc9UaGUQTpl59d3y2Y5UOGI9PjiYb1jVkxKtNXHdXnEHVdQnQWAyIro1cL4ugl0V4CezOjOy1bUT9L6qbRRs7djLbQahWCrtvjfx0l5QdmGCt2bePgGolIAJAdi04gMHV/LACqAzyR5K6dXgN8encjylnXgvhONY1ASgxivDntuc+WOIPXXwzGhG02ao/KOVBZeG3djNxb4l9EHbl3huil7WfW/+tYLlTTRpZTmwtNYHITeNr26PSGc1CWKW9/JtluU+Wrxm4LE9Bepg2EEjZ4HgP+bAAmA11IcVzFMwEcx4/tJPFsh9sm7n3X/6+CIo3GQAA'
SOURCE_SHA='47786fb7f499d6b4a34ff6e828c4dad82832fb9a77ecfbc8f961f045669ca19f'
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);d=p.read_bytes();return dict(path=str(p),bytes=len(d),sha256=hashlib.sha256(d).hexdigest())
def checked(p,want):
 v=fp(p)
 if v['sha256']!=want:raise ValueError('Actual bound input hash differs: '+str(p))
 return v
def immutable(p,value):
 if p.exists():
  if read(p)!=value:raise ValueError('Prior immutable plan differs')
 else:
  temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False)+'\n');temp.replace(p)
B.mkdir(exist_ok=True);guard=(B/'cpu-plan.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
code=gzip.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(code).hexdigest()!=SOURCE_SHA:raise ValueError('Published source bytes differ')
source=B/('peer_append_only_plan-'+COMMIT+'.py')
if source.exists():assert source.read_bytes()==code
else:source.write_bytes(code)
old=checked(A/'peer-newseed8192-g10-train-v1/plan.json','0b2212b46e2046d097f7c1a90490406ba67babf81002f15b129a61efe4dba012')
identity=checked(A/'peer-expanded-identity-v1/identity-plan.json','b2bcc13b542c3efbd90077a94138c02917940f286fde8f6ffeab2abe7542a047')
namespace={};exec(compile(code,str(source),'exec'),namespace)
plan=namespace['extend_plan'](read(old['path']),read(identity['path']))
assert len(plan['batches'])==938 and plan['historical_updates']==256 and plan['additional_updates']==682
flat=[q for batch in plan['batches'] for q in batch]
assert len(flat)==len(set(flat))==30016 and plan['batches'][:256]==read(old['path'])['batches']
if (B/'complete.json').exists():
 done=read(B/'complete.json');assert done['source_commit']==COMMIT and done['old_plan']==old and done['identity_plan']==identity
 assert read(done['plan']['path'])==plan and fp(done['plan']['path'])==done['plan']
else:
 immutable(B/'plan.json',plan)
 done=dict(status='APPEND_ONLY_OLD256_PLUS_NEW682_CUMULATIVE938_PLAN_PREPARED_NO_FITTING',source_commit=COMMIT,
  source_code=fp(source),old_plan=old,identity_plan=identity,plan=fp(B/'plan.json'),historical_questions=8192,
  new_questions=21824,total_questions=30016,historical_updates=256,new_updates=682,cumulative_updates=938,
  source_Adam=10720,target_Adam=11402,labels_or_scores_read=False,model_tensors_read=False,
  source_checkpoint_and_loss_recipe_selected=False,probe_training_submitted=False,
  actor_updates=0,final_pool_read=False,time=time.time())
 immutable(B/'complete.json',done)
print(json.dumps(dict(complete=fp(B/'complete.json'),**done),sort_keys=True))
