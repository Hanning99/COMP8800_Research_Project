#!/usr/bin/env python3
"""Verify the existing HF commit and repair the GitHub pin without data re-upload."""
import base64,getpass,hashlib,io,json,os,shutil,subprocess,sys,tempfile,urllib.request,urllib.error
from pathlib import Path
from publish_tonight import ROOT,BASE,GITHUB,HF_ID,copy_code,github_token,run,status

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 from huggingface_hub import HfApi,get_token
 cfg=json.loads((ROOT/'configs/release.json').read_text());revision=cfg['checkpoints']['revision']
 if not revision:raise RuntimeError('No completed dataset revision recorded')
 report=json.loads((BASE/'experiment/PAYLOAD_VERIFICATION.json').read_text());assert report['status']=='PAYLOADS_VERIFIED' and report['files']==71343
 tokenfile=BASE/'.upload_auth/hf_token';token=None
 if tokenfile.exists():token=tokenfile.read_text().strip();tokenfile.unlink()
 token=token or os.environ.get('HF_TOKEN') or get_token()
 if not token:token=getpass.getpass('Hugging Face token (hidden): ').strip()
 api=HfApi(token=token)
 print('Verifying the existing private Hugging Face commit; no data re-upload...',flush=True)
 info=api.repo_info(HF_ID,repo_type='dataset',revision=revision,files_metadata=True)
 if not info.private:raise RuntimeError('Unexpected public dataset')
 remote={r.rfilename:r for r in info.siblings}
 shards=json.loads((BASE/'experiment/SHARDS.json').read_text());verified=0
 for r in shards:
  f=remote.get(r['path'])
  if f is None or f.size!=r['bytes']:raise RuntimeError('Remote shard missing/size mismatch: '+r['path'])
  lfs=f.lfs
  want=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
  if want!=r['sha256']:raise RuntimeError('Remote shard SHA256 mismatch: '+r['path'])
  verified+=1
 for name in ['manifest.sqlite','BENCHMARKS.json','SOURCE_SCOPE.json','PAYLOAD_VERIFICATION.json','archive.py','README.md']:
  if name not in remote:raise RuntimeError('Remote metadata missing: '+name)
 f=remote['manifest.sqlite'];remote_sha=f.lfs.get('sha256') if isinstance(f.lfs,dict) else getattr(f.lfs,'sha256',None)
 if remote_sha and remote_sha!=digest(BASE/'experiment/manifest.sqlite'):raise RuntimeError('Remote manifest hash mismatch')
 check={'status':'PASS','dataset_revision':revision,'private':True,'remote_shards_checked':verified,'remote_compressed_bytes':sum(r['bytes'] for r in shards),'remote_shard_SHA256_agreement':True,'remote_manifest_present':True}
 (ROOT/'evaluation/results/REMOTE_DATASET_VERIFICATION.json').write_text(json.dumps(check,indent=2)+'\n')
 gh=github_token();env=dict(os.environ);env['GIT_TERMINAL_PROMPT']='0';headers={'Accept':'application/vnd.github+json','User-Agent':'COMP8800-artifact-publisher'}
 if gh:
  headers['Authorization']='Bearer '+gh;env['GIT_CONFIG_COUNT']='1';env['GIT_CONFIG_KEY_0']='http.https://github.com/.extraheader';env['GIT_CONFIG_VALUE_0']='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+gh).encode()).decode()
 try:
  with urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/Hanning99/COMP8800_Research_Project',headers=headers),timeout=30) as response:repo=json.load(response)
  if not repo.get('private'):raise RuntimeError('Unexpected public GitHub repository')
  branch=repo.get('default_branch') or 'main';owner=repo.get('owner',{})
 except urllib.error.HTTPError as exc:
  if gh or exc.code!=404:raise
  run(['git','ls-remote',GITHUB+'.git','HEAD'],env=env);branch='main';owner={}
 ready=json.loads((ROOT/'CODE_READY.json').read_text());ready['checkpoint_publication']='Uploaded privately and remotely hash-verified';ready['checkpoint_revision']=revision;(ROOT/'CODE_READY.json').write_text(json.dumps(ready,indent=2)+'\n')
 cfg['publication_status']='private dataset remotely verified; source version identified by its Git commit';(ROOT/'configs/release.json').write_text(json.dumps(cfg,indent=2)+'\n')
 print('Pinning the verified dataset revision in private GitHub...',flush=True)
 with tempfile.TemporaryDirectory(prefix='comp8800-finish-',dir=str(BASE)) as temporary:
  checkout=Path(temporary)/'repository';run(['git','clone',GITHUB+'.git',str(checkout)],env=env)
  def git(*args,check=True):return run(['git',*args],cwd=checkout,env=env,check=check).stdout.strip()
  if not git('config','user.name',check=False):git('config','user.name','Hanning Deng')
  if not git('config','user.email',check=False):git('config','user.email',str(owner.get('id',''))+'+Hanning99@users.noreply.github.com')
  copy_code(checkout);git('add','--all')
  if git('status','--porcelain'):git('commit','-m','Pin remotely verified four-benchmark checkpoint release')
  git('push','origin','HEAD:'+branch);commit=git('rev-parse','HEAD')
 receipt={'status':'PASS','scope':'four complete benchmark packages','github_private':True,'huggingface_private':True,'github_commit':commit,'github_link':GITHUB+'/tree/'+commit,'checkpoint_revision':revision,'checkpoint_link':'https://huggingface.co/datasets/'+HF_ID+'/tree/'+revision,'wattle_link':GITHUB+'/tree/'+commit,'remote_shards_SHA256_verified':verified}
 (BASE/'PUBLISH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');status(stage='complete',huggingface_uploaded=True,error=None,**receipt)
 print('\nSUCCESS — submit this fixed GitHub version to Wattle:\n'+receipt['wattle_link'],flush=True)
 print('Hugging Face fixed version: '+receipt['checkpoint_link'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  status(stage='failed_final_pin',error=type(exc).__name__+': '+str(exc));print('FINISH STOPPED:',type(exc).__name__,str(exc),file=sys.stderr)
  if isinstance(exc,subprocess.CalledProcessError) and exc.stderr:print(exc.stderr,file=sys.stderr)
  sys.exit(1)
