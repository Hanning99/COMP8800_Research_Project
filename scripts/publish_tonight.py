#!/usr/bin/env python3
"""Publish source first, then verified benchmark data, retaining private visibility."""
import argparse,base64,getpass,json,os,shutil,subprocess,sys,tempfile,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
GITHUB='https://github.com/Hanning99/COMP8800_Research_Project'
HF_ID='denghanning/COMP8800_Research_Project'
def status(**values):
 path=BASE/'UPLOAD_STATUS.json';data=json.loads(path.read_text()) if path.exists() else {}
 data.update(values);data['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());path.write_text(json.dumps(data,indent=2)+'\n')
def run(args,cwd=None,env=None,input=None,check=True):
 return subprocess.run(args,cwd=cwd,env=env,input=input,text=True,capture_output=True,check=check)
def github_token():
 token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
 if token:return token
 if shutil.which('gh'):
  result=run(['gh','auth','token'],check=False)
  if result.returncode==0:return result.stdout.strip()
 env=dict(os.environ);env['GIT_TERMINAL_PROMPT']='0'
 result=run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',env=env,check=False)
 if result.returncode==0:
  fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
  return fields.get('password')
 return None
def replace_copy(source,destination,**kwargs):
 destination=Path(destination)
 if os.path.lexists(destination):destination.unlink()
 return shutil.copy2(source,destination,**kwargs)

def copy_code(checkout):
 for item in ROOT.iterdir():
  if item.name in {'.git','.venv','__pycache__','work','outputs'}:continue
  dest=checkout/item.name
  if item.is_dir():
   def ignored(directory,names):
    excluded={n for n in names if n in {'.git','.venv','__pycache__','work','outputs'} or n.endswith('.pyc')}
    if Path(directory)==ROOT/'checkpoints' and 'local' in names:excluded.add('local')
    return excluded
   shutil.copytree(item,dest,dirs_exist_ok=True,ignore=ignored,copy_function=replace_copy)
  else:replace_copy(item,dest)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,default=BASE/'experiment');p.add_argument('--token-file',type=Path,default=BASE/'.upload_auth/hf_token');a=p.parse_args()
 ready=ROOT/'CODE_READY.json'
 print('Waiting for the focused four-benchmark code package...',flush=True)
 while not ready.exists():time.sleep(3)
 from huggingface_hub import HfApi,get_token,CommitOperationAdd
 token=None
 if a.token_file.exists():
  token=a.token_file.read_text().strip();a.token_file.unlink()
 if not token:token=os.environ.get('HF_TOKEN') or get_token()
 if not token:token=getpass.getpass('Hugging Face write token (hidden): ').strip()
 if not token:raise RuntimeError('No Hugging Face authentication supplied')
 api=HfApi(token=token)
 info=api.repo_info(HF_ID,repo_type='dataset')
 if not info.private:raise RuntimeError('Dataset is public; private publication was requested')
 gh=github_token();env=dict(os.environ);env['GIT_TERMINAL_PROMPT']='0';headers={'Accept':'application/vnd.github+json','User-Agent':'COMP8800-artifact-publisher'}
 if gh:
  headers['Authorization']='Bearer '+gh
  env['GIT_CONFIG_COUNT']='1';env['GIT_CONFIG_KEY_0']='http.https://github.com/.extraheader';env['GIT_CONFIG_VALUE_0']='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+gh).encode()).decode()
 try:
  with urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/Hanning99/COMP8800_Research_Project',headers=headers),timeout=30) as response:repo=json.load(response)
  if not repo.get('private'):raise RuntimeError('GitHub repository is public; stopped')
  branch=repo.get('default_branch') or 'main';owner=repo.get('owner',{})
 except urllib.error.HTTPError as exc:
  if gh or exc.code!=404:raise
  # A private repository can be accessible via configured Git/SSH without an API token.
  run(['git','ls-remote',GITHUB+'.git','HEAD'],env=env);branch='main';owner={}
 with tempfile.TemporaryDirectory(prefix='comp8800-publish-',dir=str(BASE)) as temporary:
  checkout=Path(temporary)/'repository';run(['git','clone',GITHUB+'.git',str(checkout)],env=env)
  def git(*args,check=True):return run(['git',*args],cwd=checkout,env=env,check=check).stdout.strip()
  if not git('config','user.name',check=False):git('config','user.name','Hanning Deng')
  if not git('config','user.email',check=False):git('config','user.email',str(owner.get('id',''))+'+'+'Hanning99@users.noreply.github.com')
  def commit_push(message):
   copy_code(checkout);git('add','--all')
   if git('status','--porcelain'):git('commit','-m',message)
   git('push','origin','HEAD:'+branch);return git('rev-parse','HEAD')
  print('1/3 Publishing source to PRIVATE GitHub...',flush=True)
  first=commit_push('Package PDR code and four benchmark reproduction entry points')
  first_link=GITHUB+'/tree/'+first;status(stage='source_uploaded_checkpoint_preparation',github_commit=first,github_link=first_link,github_private=True,huggingface_uploaded=False)
  print('SOURCE UPLOADED:',first_link,flush=True)
  print('2/3 Uploading immutable checkpoint shards as they become ready...',flush=True)
  saved=BASE/'UPLOAD_STATUS.json'
  uploaded=json.loads(saved.read_text()).get('uploaded_shards',{}) if saved.exists() else {}
  last=0
  while True:
   verification=a.archive/'PAYLOAD_VERIFICATION.json';report={}
   if verification.exists():
    try:report=json.loads(verification.read_text())
    except json.JSONDecodeError:pass
   pending=[]
   for path in sorted((a.archive/'shards').glob('*.pack')):
    rel=str(path.relative_to(a.archive));st=path.stat();identity=[st.st_size,st.st_mtime_ns]
    if uploaded.get(rel)!=identity:pending.append((path,rel,identity))
   if pending:
    batch=pending[:16]
    operations=[CommitOperationAdd(path_in_repo=rel,path_or_fileobj=str(path)) for path,rel,identity in batch]
    staged=api.create_commit(repo_id=HF_ID,repo_type='dataset',operations=operations,commit_message='Stage benchmark checkpoint shards; final manifest verification pending')
    for path,rel,identity in batch:uploaded[rel]=identity
    transferred=sum(v[0] for v in uploaded.values())
    status(stage='checkpoint_staging_upload',uploaded_shards=uploaded,uploaded_shard_count=len(uploaded),uploaded_shard_bytes=transferred,latest_partial_checkpoint_revision=staged.oid)
    print(f'Checkpoint shards uploaded: {len(uploaded)}/256; {transferred/1e9:.2f} GB',flush=True)
    continue
   if report.get('status')=='PAYLOADS_VERIFIED' and report.get('files')==71343:break
   if time.time()-last>60:print('Waiting for the next verified shard/final verification; source is already on GitHub.',flush=True);last=time.time()
   time.sleep(5)
  print('All shard bytes staged; publishing the final verified manifest and release metadata...',flush=True)
  status(stage='checkpoint_upload',archive_compressed_bytes=report['compressed_bytes'])
  result=api.upload_folder(repo_id=HF_ID,repo_type='dataset',folder_path=str(a.archive),commit_message='Four complete benchmark packages: KeyCorridor, MiniGridLoCA, DeepSea and SecureCourier',ignore_patterns=['*.tmp','*.sqlite-wal','*.sqlite-shm'])
  revision=result.oid
  final_info=api.repo_info(HF_ID,repo_type='dataset',revision=revision)
  if not final_info.private:raise RuntimeError('Unexpected dataset visibility change')
  cfgpath=ROOT/'configs/release.json';cfg=json.loads(cfgpath.read_text());cfg['checkpoints']['revision']=revision;cfg['publication_status']='private checkpoint revision pinned; source version identified by its Git commit';cfgpath.write_text(json.dumps(cfg,indent=2)+'\n')
  shutil.copy2(verification,ROOT/'evaluation/results/ARCHIVE_INTEGRITY.json')
  print('3/3 Pinning the dataset revision in GitHub...',flush=True)
  final=commit_push('Pin verified four-benchmark checkpoint dataset revision')
  receipt={'status':'PASS','scope':'four complete benchmark packages','github_private':True,'huggingface_private':True,'github_commit':final,'github_link':GITHUB+'/tree/'+final,'checkpoint_revision':revision,'checkpoint_link':'https://huggingface.co/datasets/'+HF_ID+'/tree/'+revision,'wattle_link':GITHUB+'/tree/'+final}
  (BASE/'PUBLISH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');status(stage='complete',**receipt)
  print('\nSUCCESS — both repositories remain private. Wattle submission link:\n'+receipt['wattle_link'],flush=True)
  print('Checkpoint revision: '+revision,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  status(stage='failed',error=type(exc).__name__+': '+str(exc));print('UPLOAD STOPPED:',type(exc).__name__,str(exc),file=sys.stderr)
  if isinstance(exc,subprocess.CalledProcessError) and exc.stderr:print(exc.stderr,file=sys.stderr)
  sys.exit(1)
