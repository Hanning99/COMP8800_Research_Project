"""Check copies, checkpoint readability, schedules, saved policies, and all final actors on CPU."""
from pathlib import Path
import argparse,hashlib,json,os,sys
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
os.environ['MINIGRID_LOCA_PROTOCOL']=str(ROOT/'inputs/loca_exact/PROTOCOL_LOCK.json')
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
torch.set_num_threads(1);torch.set_num_interop_threads(1);BLAS=threadpool_limits(limits=1)
from run_study import Actor,schedules,GAMMA,extract_policy
from minigrid_loca_core import build_graph,evaluate_policy

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    copies=json.loads((ROOT/'provenance/SOURCE_COPY_MANIFEST.json').read_text());cache={};checked=0
    for row in copies:
        path=ROOT/row['path'];assert path.is_file() and not path.is_symlink()
        if row['path'] in ['src/common.py','src/learner.py']:continue
        key=(path.stat().st_dev,path.stat().st_ino)
        if key not in cache:cache[key]=sha(path)
        assert cache[key]==row['sha256'],row['path'];checked+=1
    checkpoints=list(ROOT.glob('runs/**/*.pt'))+list(ROOT.glob('starts/**/*.pt'))+list(ROOT.glob('inputs/**/*.pt'))
    ckcache=set()
    for path in checkpoints:
        key=(path.stat().st_dev,path.stat().st_ino)
        if key in ckcache:continue
        z=torch.load(path,map_location='cpu',weights_only=False);assert isinstance(z,dict);ckcache.add(key)
    print(f'Hash checks: {checked}; checkpoint paths: {len(checkpoints)}',flush=True)
    g=build_graph();states=torch.arange(g.n_states);actor=Actor().eval();runs=json.loads((ROOT/'RUNS.json').read_text())
    starts={};reports=[];maxerr=0.;snapshoterror=0.;policyerror=0.;nsnapshots=0
    for i,r in enumerate(runs):
        if r['common_initial'] not in starts:
            z=torch.load(ROOT/r['common_initial'],map_location='cpu',weights_only=False)
            schedule=schedules(g,r['seed'])
            h=hashlib.sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(schedule.items()))).hexdigest()
            assert h==z['schedule_sha256']
            for rel,want in z['inputs'].items():assert sha(ROOT/rel)==want,(rel,want)
            starts[r['common_initial']]=h
        curve=pd.read_csv(ROOT/r['curve']).set_index('update')
        final=torch.load(ROOT/r['final'],map_location='cpu',weights_only=False)
        assert final['completed_updates']==4096
        state=final.get('state',final);actor.load_state_dict(state['models'][0])
        policy=extract_policy(actor,states)
        metrics=evaluate_policy(g,policy,GAMMA)
        error=max(abs(v-float(curve.loc[4096,k])) for k,v in metrics.items())
        assert error<1e-5,(r['directory'],error);maxerr=max(maxerr,error)
        for row in final['curves']:
            assert max(abs(float(row[k])-float(curve.loc[row['update'],k])) for k in metrics)<1e-12
        stored=[]
        if r['method']=='PDR':
            for path in sorted((ROOT/r['directory']/'snapshots').glob('update_*.npz')):
                with np.load(path) as z:stored.append((int(path.stem.split('_')[1]),z['full_policy']))
        elif (ROOT/r['directory']/'POLICIES.npz').exists():
            with np.load(ROOT/r['directory']/'POLICIES.npz') as z:stored=list(zip(z['updates'],z['policy']))
        for u,pol in stored:
            sm=evaluate_policy(g,pol,GAMMA)
            err=max(abs(v-float(curve.loc[u,k])) for k,v in sm.items())
            assert err<1e-12,(r['directory'],u,err);snapshoterror=max(snapshoterror,err);nsnapshots+=1
            if u==4096:policyerror=max(policyerror,float(np.max(np.abs(policy-pol))))
        reports.append(dict(backbone=r['backbone'],method=r['method'],seed=r['seed'],final_return_error=error))
        if (i+1)%40==0:print(f'Final actors checked: {i+1}/{len(runs)}',flush=True)
    report=dict(status='PASS',source_file_hashes_checked=checked,checkpoint_paths=len(checkpoints),unique_checkpoints_loaded=len(ckcache),
        complete_initial_states=len(starts),final_actors_reconstructed=len(runs),maximum_CPU_final_return_error=maxerr,
        saved_policies_evaluated=nsnapshots,maximum_saved_policy_metric_error=snapshoterror,maximum_CPU_policy_error=policyerror,
        neural_only=True,details=reports,scope='All archived final actors evaluated on CPU; this does not rerun all training trajectories.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))

if __name__=='__main__':main()
