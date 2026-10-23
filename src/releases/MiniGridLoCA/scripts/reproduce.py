"""Replay selected neural-reference runs using the archived numerical implementation."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
KEYS=['models','optimizers','rng_torch','rng_cuda','rng_numpy','rng_python']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backbone',required=True,choices=['AWAC','SAC','QAC'])
    p.add_argument('--method',required=True,choices=['TD','PER','DISCOR','EXTRA_TD','IQL','CQL','PS','PDR'])
    p.add_argument('--seed',type=int)
    p.add_argument('--updates',type=int,default=4096)
    p.add_argument('--mode',choices=['replay','final','early'],default='replay')
    p.add_argument('--resume',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tolerance',type=float,default=1e-5)
    a=p.parse_args()
    import torch,numpy as np,pandas as pd
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for archived training arithmetic. CPU final-actor audit: scripts/verify.py.')
    if torch.cuda.device_count()!=1:raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one GPU.')
    from common import state_equal
    from hypotheses import HypothesisLearner
    from baselines import BaselineLearner
    runs=json.loads((ROOT/'RUNS.json').read_text())
    selected=[r for r in runs if r['backbone']==a.backbone and r['method']==a.method and (a.seed is None or r['seed']==a.seed)]
    if not selected:raise ValueError('Run not included in this release')
    r=selected[0];cell=r['cell'];seed=r['seed'];config=r['config']
    if a.out.exists():raise FileExistsError('Use a new output directory; frozen evidence is never overwritten.')
    if not 0<=a.updates<=4096:raise ValueError('updates must be between 0 and 4096')
    if a.mode=='early' and not r['early']:raise ValueError('Only CQL/IQL have archived EARLY checkpoints')
    l=HypothesisLearner(cell,seed) if r['method']=='PDR' else BaselineLearner(cell,seed,config)
    common=torch.load(ROOT/r['common_initial'],map_location='cpu',weights_only=False)
    base=l.state() if r['method']=='PDR' else l.base_initial
    assert state_equal(base,{k:common[k] for k in KEYS}),'Constructed common state mismatch'
    schedule_hash=hashlib.sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(l.schedule.items()))).hexdigest()
    assert schedule_hash==common['schedule_sha256'],'Schedule differs from original'
    initial=torch.load(ROOT/r['initial'],map_location='cpu',weights_only=False)
    initial=initial.get('state',initial)
    if r['method']=='PDR':initial={k:initial[k] for k in KEYS}
    elif 'recipe' not in initial:initial['recipe']=copy.deepcopy(config)
    if r['engine']=='tuned':l.tau=config['target_tau']
    l.restore(initial)
    assert state_equal(l.state(),initial),'Initial restore mismatch'
    start=0
    if a.resume:
        z=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert (z['cell']['id'],z['seed'],z['method'],z['config'])==(cell['id'],seed,a.method,config)
        assert z['schedule_sha256']==schedule_hash
        l.restore(z['state']);start=z['completed_updates']
    elif a.mode in ['final','early']:
        z=torch.load(ROOT/r[a.mode],map_location='cpu',weights_only=False)
        l.restore(z if a.method=='PDR' else z['state']);start=z['completed_updates']
    end=start if a.mode!='replay' else a.updates
    assert end>=start
    a.out.mkdir(parents=True)
    curves=[];policies=[];measured=[];tick=time.perf_counter()
    def measure(u):
        curves.append(dict(update=u,**l.evaluate()));policies.append(l.policy());measured.append(u)
    measure(start)
    for u in range(start,end):
        if a.method=='PDR':l.step(u,config['arm'])
        else:l.step(u)
        if u+1 in l.checkpoints or u+1==end:measure(u+1)
    torch.cuda.synchronize()
    state=l.state();policy=l.policy()
    checkpoint=dict(state=state,cell=cell,seed=seed,method=a.method,config=config,completed_updates=end,schedule_sha256=schedule_hash)
    torch.save(checkpoint,a.out/'CHECKPOINT.pt')
    l.restore(torch.load(a.out/'CHECKPOINT.pt',map_location='cpu',weights_only=False)['state'])
    assert state_equal(state,l.state()) and np.array_equal(l.policy(),policy),'Serialization changed learner state'
    pd.DataFrame(curves).to_csv(a.out/'LEARNING_CURVE.csv',index=False)
    np.savez_compressed(a.out/'POLICIES.npz',updates=measured,policy=policies)
    original=pd.read_csv(ROOT/r['curve']).set_index('update')
    errors=[];policy_errors=[];tested=[]
    for row,policy in zip(curves,policies):
        u=row['update']
        if u not in original.index:continue
        errors.append(max(abs(float(row[k])-float(original.loc[u,k])) for k in l.evaluate() if k in original.columns));tested.append(u)
        if a.method=='PDR':
            stored=np.load(ROOT/r['directory']/'snapshots'/f'update_{u:05d}.npz')['full_policy']
            policy_errors.append(float(np.max(np.abs(stored-policy))))
        elif (ROOT/r['directory']/'POLICIES.npz').exists():
            with np.load(ROOT/r['directory']/'POLICIES.npz') as z:
                stored=z['policy'][list(z['updates']).index(u)]
            policy_errors.append(float(np.max(np.abs(stored-policy))))
    maxerror=max(errors,default=0.)
    report=dict(status='PASS' if maxerror<=a.tolerance else 'FAIL',backbone=a.backbone,method=a.method,seed=seed,
        mode=a.mode,resumed_from=str(a.resume) if a.resume else None,updates=end,start=start,checked_updates=tested,
        maximum_metric_error=maxerror,maximum_saved_policy_error=max(policy_errors,default=None),
        metric_tolerance=a.tolerance,serialization_bitwise_equal=True,schedule_sha256=schedule_hash,
        common_start_bitwise_equal=True,elapsed_seconds=time.perf_counter()-tick,gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,numpy=np.__version__)
    (a.out/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    assert report['status']=='PASS',report

if __name__=='__main__':main()
