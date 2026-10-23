"""Download an immutable private dataset snapshot; tokens come from the environment."""
import argparse,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'checkpoints/local');p.add_argument('--revision');a=p.parse_args()
 release=json.loads((ROOT/'configs/release.json').read_text())['checkpoints'];revision=a.revision or release['revision']
 if not revision:raise RuntimeError('No checkpoint revision published yet. Set configs/release.json after uploading; do not evaluate an unspecified mutable branch.')
 from huggingface_hub import snapshot_download
 snapshot_download(repo_id=release['repo_id'],repo_type='dataset',revision=revision,local_dir=str(a.out),token=os.environ.get('HF_TOKEN'))
 print('Downloaded pinned checkpoint release to',a.out)
if __name__=='__main__':main()
