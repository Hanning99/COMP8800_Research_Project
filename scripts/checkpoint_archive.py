#!/usr/bin/env python3
"""List, verify or restore the independent SHA256 + zstd experiment archive."""
import argparse,hashlib,json,os,shutil,sqlite3,sys,io
from pathlib import Path
import zstandard as zstd
ROOT=Path(__file__).resolve().parent

class BoundedReader:
 def __init__(self,path,offset,length):self.file=path.open('rb');self.file.seek(offset);self.remaining=length
 def read(self,n=-1):
  n=self.remaining if n<0 else min(n,self.remaining)
  data=self.file.read(n);self.remaining-=len(data);return data
 def readable(self):return True
 def close(self):self.file.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()

def safe(root,name):
 p=root/name
 if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError(name)
 if not p.resolve().is_relative_to(root.resolve()):raise ValueError('Path escapes destination: '+name)
 return p

def main():
 global ROOT
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('action',choices=['list','verify','restore'])
 ap.add_argument('--archive-root',type=Path,help='Checkpoint dataset directory; defaults to this script directory')
 ap.add_argument('--prefix',action='append',default=[],help='Package-relative source prefix; repeat for multiple roots')
 ap.add_argument('--release',choices=['KeyCorridor','MiniGridLoCA','DeepSea','SecureCourier'])
 ap.add_argument('--output',type=Path,help='New restore directory; never overwrite existing data')
 a=ap.parse_args()
 if a.archive_root:ROOT=a.archive_root.resolve()
 prefixes=a.prefix[:]
 if a.release:prefixes.append('AAMAS-Experiment/Effectiveness/'+a.release)
 db=sqlite3.connect('file:'+str(ROOT/'manifest.sqlite')+'?mode=ro',uri=True)
 if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise RuntimeError('Manifest database is corrupt')
 db.row_factory=sqlite3.Row
 selected={r['path']:dict(r) for r in db.execute('SELECT * FROM files') if not prefixes or any(r['path']==p or r['path'].startswith(p.rstrip('/')+'/') for p in prefixes)}
 if not selected:raise RuntimeError('No matching files')
 select_all=not prefixes
 links=json.loads((ROOT/'LINKS.json').read_text())
 chosen=[]
 # Close selected directory links recursively before restoring any bytes.
 for _ in range(len(links)+1):
  changed=False
  for link in links:
   name=link.get('archive_path');target=link.get('archive_target')
   if not name or not target or link in chosen:continue
   if select_all or any(name==p or name.startswith(p.rstrip('/')+'/') for p in prefixes):
    chosen.append(link);prefixes.append(target);changed=True
    for r in db.execute('SELECT * FROM files WHERE path=? OR path LIKE ?', (target,target+'/%')):selected[r['path']]=dict(r)
  if not changed:break
 unique={r['sha256']:r for r in selected.values()}
 restore_groups={(r['sha256'],r['mode'],r['mtime_ns']):r['bytes'] for r in selected.values()}
 summary={'estimated_restored_bytes':sum(restore_groups.values()),'files':len(selected),'checkpoint_candidate_paths':sum(r['checkpoint_candidate'] for r in selected.values()),'unique_objects':len(unique),'logical_bytes':sum(r['bytes'] for r in selected.values()),'unique_bytes':sum(r['bytes'] for r in unique.values()),'links':len(chosen)}
 print(json.dumps(summary),flush=True)
 if a.action=='list':return
 if a.action=='restore':
  if a.output is None:ap.error('--output is required')
  out=a.output.absolute()
  if out.exists():raise FileExistsError(out)
  out.parent.mkdir(parents=True,exist_ok=True)
  if shutil.disk_usage(out.parent).free<summary['estimated_restored_bytes']+2**30:raise RuntimeError('Not enough space for independent restored files')
  out.mkdir();restored={}
 for i,(sha,r) in enumerate(unique.items()):
  blob=ROOT/'objects'/sha[:2]/(sha+'.zst')
  if blob.exists():source=blob.open('rb')
  else:
   record=db.execute('SELECT shard,offset,length FROM objects WHERE sha256=?',(sha,)).fetchone()
   if record is None:raise FileNotFoundError(sha)
   source=BoundedReader(ROOT/record[0],record[1],record[2])
  h=hashlib.sha256();count=0;dest=None;writer=None
  if a.action=='restore':
   dest=safe(out,r['path']);dest.parent.mkdir(parents=True,exist_ok=True);writer=dest.open('xb')
  try:
   with source as f,zstd.ZstdDecompressor().stream_reader(f) as stream:
    for b in iter(lambda:stream.read(8<<20),b''):
     h.update(b);count+=len(b)
     if writer:writer.write(b)
  finally:
   if writer:writer.close()
  if h.hexdigest()!=sha or count!=r['bytes']:raise RuntimeError('Payload verification failed: '+sha)
  if dest:
   dest.chmod(r['mode']);os.utime(dest,ns=(r['mtime_ns'],r['mtime_ns']));restored[sha]=dest
  if (i+1)%2000==0:print('Verified objects:',i+1,'/',len(unique),flush=True)
 if a.action=='restore':
  copies={(sha,r['mode'],r['mtime_ns']):restored[sha] for sha,r in unique.items()}
  for r in selected.values():
   first=restored[r['sha256']];dest=safe(out,r['path'])
   if dest==first:continue
   dest.parent.mkdir(parents=True,exist_ok=True)
   key=(r['sha256'],r['mode'],r['mtime_ns'])
   # Deduplicate only files with matching content and metadata within this tree.
   if key in copies:os.link(copies[key],dest)
   else:
    shutil.copyfile(first,dest);dest.chmod(r['mode']);os.utime(dest,ns=(r['mtime_ns'],r['mtime_ns']));copies[key]=dest
  for link in sorted(chosen,key=lambda x:len(Path(x['archive_path']).parts)):
   dst=safe(out,link['archive_path']);target=safe(out,link['archive_target'])
   dst.parent.mkdir(parents=True,exist_ok=True)
   if os.path.lexists(dst):raise FileExistsError(dst)
   dst.symlink_to(os.path.relpath(target,dst.parent))
  (out/'RESTORE_REPORT.json').write_text(json.dumps({'status':'PASS',**summary,'source_archive':str(ROOT)},indent=2)+'\n')
 print('PASS:',a.action,flush=True)
if __name__=='__main__':main()
