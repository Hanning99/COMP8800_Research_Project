"""Check independent file hashes, all checkpoint contents and stored trajectories."""
from pathlib import Path
import argparse, hashlib, json
ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--skip-checkpoints',action='store_true');a=p.parse_args()
    manifest=json.loads((ROOT/'provenance/COPIED_FILES.json').read_text());cache={};total=0
    for name,record in manifest.items():
        path=ROOT/name;assert path.is_file() and not path.is_symlink(),name
        st=path.stat();key=(st.st_dev,st.st_ino)
        if key not in cache:cache[key]=digest(path)
        assert cache[key]==record['sha256'],name;total+=st.st_size
    print('Copied-file hashes verified:',len(manifest),flush=True)
    runs=json.loads((ROOT/'RUNS.json').read_text());assert len(runs)==640
    error=0.;count=0
    if not a.skip_checkpoints:
        import torch
        import pandas as pd
        import numpy as np
        checked=set()
        for i,r in enumerate(runs):
            curve=pd.read_csv(ROOT/r['curve'])
            assert digest(ROOT/r['curve'])==r['curve_sha256']
            assert digest(ROOT/r['final'])==r['final_sha256']
            for path in [r['initial'],r['common_initial'],r['early'],r['final']]:
                if not path or path in checked:continue
                checked.add(path);z=torch.load(ROOT/path,map_location='cpu',weights_only=False)
                state=z.get('state',z)
                for key in ['models','optimizers','rng_torch','rng_cuda','rng_numpy','rng_python']:assert key in state,(path,key)
                assert len(state['models'])>=3 and len(state['optimizers'])>=2
                if 'curves' in z:
                    stored=pd.DataFrame(z['curves']);paired=stored.merge(curve,on='update',suffixes=('_a','_b'),validate='one_to_one')
                    assert len(paired)==len(stored)
                    for key in set(stored)&set(curve)-{'update'}:
                        diff=float(np.max(np.abs(paired[key+'_a']-paired[key+'_b'])));error=max(error,diff)
                        assert diff<1e-12,(path,key,diff)
                if path==r['final']:assert z['completed_updates']==512
                count+=1
            if i%80==0:print(f'Checkpoint audit {i}/640',flush=True)
        # Also inspect ancillary native/common starts and the four neural
        # anchor checkpoints, not just the files used directly by the runner.
        for path in sorted(name for name in manifest if name.endswith('.pt') and name not in checked):
            z=torch.load(ROOT/path,map_location='cpu',weights_only=False)
            assert isinstance(z,dict) and z,path
            state=z.get('state',z)
            if 'models' in state:
                for key in ['optimizers','rng_torch','rng_cuda','rng_numpy','rng_python']:assert key in state,(path,key)
            else:
                assert 'critic' in state,('Unknown checkpoint schema',path)
                assert all(torch.isfinite(v).all() for v in state['critic'].values() if torch.is_tensor(v)),path
            count+=1
    report=dict(status='PASS',copied_paths=len(manifest),independent_payload_files=len(cache),logical_gib=total/2**30,
        runs=640,checkpoints_loaded=count,max_stored_curve_error=error,scope='Integrity and stored-state/CSV agreement; GPU replay reported separately')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
