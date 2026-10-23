"""Verify payload hashes, all checkpoint contents, exact references and curves."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
import torch
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'vendor'))
import deepsea_experiment as core
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def soft_q(pi,g,entropy):
    pi=np.asarray(pi,np.float64);v=np.zeros(len(pi));q=np.zeros_like(pi)
    for t in range(int(g['size'])-1,-1,-1):
        ids=np.flatnonzero(g['states'][:,0]==t);q[ids]=g['reward'][ids]
        if t<int(g['size'])-1:q[ids]+=v[g['next'][ids]]
        v[ids]=(pi[ids]*(q[ids]-entropy*np.log(np.clip(pi[ids],1e-30,1)))).sum(1)
    return q
def check_state(z):
    for k in ['models','optimizers','rng_torch','rng_cuda','rng_numpy','rng_python']:assert k in z,k
    for model in z['models']:
        assert model
        for t in model.values():assert torch.isfinite(t).all()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((R/'provenance/COPIED_FILES.json').read_text());cache={}
    assert not any(p.is_symlink() for p in R.rglob('*'))
    for rel,z in manifest.items():
        p=R/rel;s=p.stat();key=(s.st_dev,s.st_ino)
        if key not in cache:cache[key]=sha(p)
        assert cache[key]==z['sha256'],rel
    print('Copied hashes verified:',len(manifest),flush=True)
    records=json.loads((R/'RUNS.json').read_text());checked=set();worlds={};referror=0.;curveerror=0.;policyerror=0.;initials={}
    for i,r in enumerate(records):
        n,seed=r['N'],r['seed'];key=(n,seed)
        if key not in worlds:
            w=R/'inputs/deepsea/worlds'/f'N{n}'/f'seed_{seed}'
            g=dict(np.load(w/'graph.npz'));common=torch.load(w/'COMMON_START.pt',map_location='cpu',weights_only=False)
            assert all(k in common for k in ['actor','critic','target','actor_opt','critic_opt','pi0','q0'])
            for name in ['actor','critic','target']:
                assert all(torch.isfinite(t).all() for t in common[name].values())
            checked.add(str((w/'COMMON_START.pt').relative_to(R)))
            ref=torch.load(R/r['exact_reference'],map_location='cpu',weights_only=False);checked.add(r['exact_reference'])
            assert np.array_equal(common['pi0'],ref['policy']) and np.array_equal(common['q0'],ref['qold'])
            assert ref['entropy']==.001
            referror=max(referror,float(np.max(np.abs(soft_q(ref['policy'],g,ref['entropy'])-ref['soft_qold']))),float(np.max(np.abs(core.exact_q(ref['policy'],g)[1]-ref['qold']))))
            schedule=dict(np.load(w/'schedule.npz'))
            sh=hashlib.sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(schedule.items()))).hexdigest();assert sh==r['schedule_sha256']
            # Verify every saved transition against the official bsuite implementation.
            audit=core.exhaustive_source_audit(n,seed,g);assert all(x['repeatable'] for x in audit)
            worlds[key]=g
        g=worlds[key]
        for name in ['base_initial','initial']:
            rel=r[name]
            if rel not in checked:check_state(torch.load(R/rel,map_location='cpu',weights_only=False));checked.add(rel)
        z=torch.load(R/r['final'],map_location='cpu',weights_only=False);checked.add(r['final']);check_state(z['state']);assert z['update']==8192
        assert z['schedule_sha256']==r['schedule_sha256']
        curve=pd.read_csv(R/r['curve']);stored=pd.DataFrame(z['curves'])
        assert np.array_equal(curve['update'],stored['update'])
        for col in curve.columns:curveerror=max(curveerror,float(np.max(np.abs(curve[col]-stored[col]))))
        for name in ['models','optimizers']:assert len(z['state'][name])>0
        with np.load(R/Path(r['curve']).parent/'POLICIES.npz') as policies:
            assert np.array_equal(policies['updates'],curve['update'])
            for (_,row),pi in zip(curve.iterrows(),policies['policy']):
                values=core.exact_metrics(pi,g)
                for k,v in values.items():policyerror=max(policyerror,abs(v-row[k]))
        initials.setdefault(key,[]).append(float(curve['expected_return'].iloc[0]))
        if i%80==0:print('Checked runs:',i,flush=True)
    initialerror=max(max(v)-min(v) for v in initials.values());assert all(len(v)==8 for v in initials.values())
    allpt={str(p.relative_to(R)) for p in R.rglob('*.pt')};assert checked==allpt,(allpt-checked,checked-allpt)
    assert max(referror,curveerror,policyerror,initialerror)<1e-12
    report=dict(status='PASS',copied_paths=len(manifest),runs=len(records),checkpoint_files_loaded=len(checked),
        exact_references_checked=len(worlds),official_bsuite_graphs_checked=len(worlds),max_reference_error=referror,
        max_stored_curve_error=curveerror,max_saved_policy_metric_error=policyerror,max_paired_initial_return_difference=initialerror,
        scope='All payload hashes/checkpoints, all exact reference tables, official environment transition audits, and every saved policy/CSV agreement; GPU training replay is verified separately.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
