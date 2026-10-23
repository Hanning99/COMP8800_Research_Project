"""CPU reconstruction of every final actor; allow CPU/GPU rounding only."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import torch
sys.dont_write_bytecode=True
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root;sys.path.insert(0,str(root/'vendor'));import deepsea_experiment as core
    core.DEVICE=torch.device('cpu');torch.set_num_threads(1)
    rows=[];graphs={}
    for i,r in enumerate(json.loads((root/'RUNS.json').read_text())):
        key=(r['N'],r['seed'])
        if key not in graphs:graphs[key]=dict(np.load(root/'inputs/deepsea/worlds'/f'N{key[0]}'/f'seed_{key[1]}'/'graph.npz'))
        g=graphs[key];z=torch.load(root/r['final'],map_location='cpu',weights_only=False)
        actor=core.Actor(g['features'].shape[1]);actor.load_state_dict(z['state']['models'][0])
        pi=core.actor_policy(actor,g)
        with np.load(root/Path(r['curve']).parent/'POLICIES.npz') as source:expected=source['policy'][-1];assert source['updates'][-1]==8192
        policyerror=float(np.max(np.abs(pi-expected)))
        expected_return=float(pd.read_csv(root/r['curve']).expected_return.iloc[-1]);actual=core.exact_metrics(pi,g)['expected_return']
        error=abs(actual-expected_return);assert max(policyerror,error)<1e-5,(r['N'],r['seed'],r['method'],policyerror,error)
        rows.append(dict(N=r['N'],seed=r['seed'],method=r['method'],max_policy_error=policyerror,return_error=error))
        if i%80==0:print('Reconstructed final actors:',i,flush=True)
    report=dict(status='PASS',final_actors_reconstructed=len(rows),device='CPU',cpu_gpu_tolerance=1e-5,
        max_policy_error=max(r['max_policy_error'] for r in rows),max_return_error=max(r['return_error'] for r in rows),runs=rows)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='runs'}),flush=True)
if __name__=='__main__':main()
