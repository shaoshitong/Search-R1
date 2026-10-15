"""One read-only validation DAG per recovery boundary; no shared module patches.

Each original pinned consumer still executes its complete checks. Only repeated
calls to the same immutable handoff inside this boundary reuse a completed
proof. Sealing reruns every original consumer, revalidates full publications and
rehashes retained bytes. A stage is never reused by a later recovery boundary.
Host, process, accounting and payload queries are never memoized.
"""
import copy
import functools
import json
from pathlib import Path
import stat

from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader

RECOVERIES={
    '09ccff26c5b764cc4c6a285c1e8f57a8e9b07726':'recover_sigma3_archive5',
    'f979d53cdb0d348e87c5877d84403252e85cd784':'recover_sigma3_followthrough_service',
    'dc9c79bc5e5c7d1b4e3134d30fe168223de10215':'recover_sigma3_archive50',
    'b89495c8ec1f2a7fb0d5cd70d89974c16c801cff':'recover_sigma3_archive50_second',
    '4194ad2c21c5e10ae5ee3b0212fb6fbf3e02f41b':'recover_sigma3_archive100',
    '23ab8b0c8af403ec830644ff4fa04eeef7f55927':'recover_sigma3_archive100_second',
}


def need(ok,message):reader.need(ok,'LINEAGE_STAGE: '+message)


class Stage:
    """Private imported instances and proofs, discarded after one seal."""
    def __init__(self):
        self.modules={};self.publications={};self.prepared=set();self.controls=set()
        self.proofs={};self.calls={};self.order=[];self.running=set();self.retained={}
        self.state='open'

    def __getattr__(self,name):
        # Ordinary live sensors and all original byte readers remain original.
        return getattr(reader,name)

    def remember(self,path,sha,publication_sha=None):
        path=Path(path);s=path.lstat()
        need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and
             stat.S_ISREG(s.st_mode) and s.st_uid==reader.OWNER,'unsafe retained evidence')
        row=dict(path=str(path),sha256=sha,bytes=s.st_size,owner=s.st_uid,
                 publication_sha256=publication_sha)
        old=self.retained.get(str(path))
        need(old is None or all(old[k]==row[k] for k in ('path','sha256','bytes','owner')),
             'evidence changed inside validation stage: '+str(path))
        self.retained[str(path)]=row

    def read_file(self,path,**kwargs):
        value,record=reader.read_file(path,**kwargs)
        self.remember(record['path'],record['sha256'])
        return value,record

    def document(self,path):
        value,record=reader.document(path)
        self.remember(record['path'],record['sha256'])
        return value,record

    def prepare_control(self,control,proof):
        if id(control) in self.controls:return
        self.controls.add(id(control));original=control.digest
        @functools.wraps(original)
        def digest(path):
            value=original(path);self.remember(path,value,proof['publication_sha256']);return value
        # load_fixed returned a private controller, never an active/shared one.
        control.digest=digest

    def prepare(self,module,commit,name):
        if id(module) in self.prepared:return
        self.prepared.add(id(module))
        if hasattr(module,'reader'):module.reader=self
        if callable(getattr(module,'consume',None)):
            original=module.consume
            @functools.wraps(original)
            def consume(*args,**kwargs):
                key=(id(module),json.dumps([args,kwargs],sort_keys=True,allow_nan=False))
                need(self.state!='sealed','closed stage cannot be reused')
                if key in self.proofs:return copy.deepcopy(self.proofs[key])
                need(self.state=='open' and key not in self.running,'new/cyclic dependency during stage seal')
                need(args and args[0]==commit,'consumer commit differs from its private publication')
                self.running.add(key)
                try:value=original(*args,**kwargs)
                finally:self.running.remove(key)
                self.proofs[key]=copy.deepcopy(value);self.calls[key]=(original,args,kwargs)
                self.order.append(key)
                return copy.deepcopy(value)
            module.consume=consume
        if callable(getattr(module,'service',None)):
            factory=module.service
            @functools.wraps(factory)
            def service():
                value=factory()
                # The factory itself creates a private sigma3_service_recovery.
                self.prepare(value,commit,name+'.service')
                return value
            module.service=service

    def load_fixed(self,commit,filename,control_attribute):
        key=(commit,filename,control_attribute)
        need(self.state!='sealed','closed stage cannot load modules')
        if key not in self.modules:
            need(self.state=='open','new source dependency appeared during seal')
            module,proof=reader.load_fixed(commit,filename,control_attribute)
            need(proof['commit']==commit,'loaded publication commit differs')
            old=self.publications.get(commit)
            need(old is None or old==proof,'publication changed inside validation stage')
            self.publications[commit]=proof
            self.remember(proof['publication_path'],proof['publication_sha256'])
            self.prepare_control(getattr(module,control_attribute),proof)
            if RECOVERIES.get(commit)==filename:self.prepare(module,commit,filename)
            self.modules[key]=(module,proof)
        return self.modules[key]

    def seal(self):
        need(self.state=='open' and self.proofs,'stage must have a completed dependency graph')
        self.state='sealing'
        try:
            # Dependencies finished first when the original recursive call ran.
            # Each original function now repeats its own byte/live checks once.
            for key in tuple(self.order):
                function,args,kwargs=self.calls[key]
                need(function(*args,**kwargs)==self.proofs[key],'original handoff changed before stage seal')
            for commit,proof in self.publications.items():
                _,actual=reader.publication(commit)  # Full set AND every source SHA.
                need(actual==proof,'full source publication changed before seal')
            for row in tuple(self.retained.values()):
                _,actual=reader.read_file(row['path'],expected_sha=row['sha256'],expected_bytes=row['bytes'])
                need(Path(actual['path']).stat().st_uid==row['owner'],'retained owner changed')
        finally:
            self.state='sealed'


def consume(commit,filename):
    need(RECOVERIES.get(commit)==filename,'unregistered top-level recovery consumer')
    stage=Stage();module,_=stage.load_fixed(commit,filename,'control')
    value=module.consume(commit);stage.seal()
    return value
