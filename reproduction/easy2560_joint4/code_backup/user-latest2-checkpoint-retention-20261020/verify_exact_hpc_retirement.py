import pathlib,json,hashlib,stat,time,os
root=pathlib.Path("/tmp/easy-authorized-latest2-eebf738cb45d434cb9cebb167145cf6f");raw=(root/'plan.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=="2282bd9a2c8d6be8368fa1042c9d24450ba55346da7b39d21be30f649e9114e4"
p=json.loads(raw);r=json.loads((root/'result.json').read_text());assert r['deleted_file_count']==52 and r['all_target_paths_absent'] and r['all_preserved_payload_metadata_unchanged']
assert all(not pathlib.Path(x['path']).exists() for x in p['delete_files'])
for x in p['preserved_payload_members']:
 q=pathlib.Path(x['path']);s=q.lstat();assert stat.S_ISREG(s.st_mode) and (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(x['device'],x['inode'],x['bytes'],x['mtime_ns'])
for g in p['groups']:
 for q in g['kept_checkpoint_dirs']:
  q=pathlib.Path(q)
  assert any(q.rglob('*.safetensors'))
  if g['name'].startswith('TOPD'):assert (q/'resume-complete.json').is_file()
assert not (root/'failure.json').exists()
print(json.dumps(dict(checked_unix=time.time(),plan_sha256="2282bd9a2c8d6be8368fa1042c9d24450ba55346da7b39d21be30f649e9114e4",result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),deleted_paths_still_absent=52,preserved_payload_members_unchanged=len(p['preserved_payload_members']),all_latest_checkpoint_markers_present=True,kept_steps={g['name']:g['kept_steps'] for g in p['groups']},confirmed_last_link_unlinked_allocated_bytes=r['confirmed_last_link_unlinked_allocated_bytes'],read_only=True,source_code_logs_data_cache_modified=False)))
