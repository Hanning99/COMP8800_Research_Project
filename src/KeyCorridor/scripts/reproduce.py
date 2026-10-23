"""Portable replay of selected neural runs, using unchanged numerical code."""
from pathlib import Path
import argparse, copy, hashlib, json, os, subprocess, sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/keycorridor-release-mpl')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['S3R2','S3R3','S4R3','S5R3'],required=True)
    p.add_argument('--method',choices=['TD','PER','DISCOR','EXTRA_TD','IQL','CQL','PS','PDR'],required=True)
    p.add_argument('--seed',type=int,help='Default: first archived seed for this task/method')
    p.add_argument('--mode',choices=['replay','final','resume'],default='replay')
    p.add_argument('--updates',type=int,default=512)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tolerance',type=float,default=1e-5)
    a=p.parse_args()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for unchanged archived training implementation')
    if torch.cuda.device_count()!=1: raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one GPU')
    from common import state_equal
    from emphasis import EmphasisLearner
    from baselines import BaselineLearner
    import numpy as np
    import pandas as pd
    runs=json.loads((ROOT/'RUNS.json').read_text())
    matches=[r for r in runs if r['cell']['task']==a.task and r['method']==a.method and (a.seed is None or r['seed']==a.seed)]
    if not matches: raise ValueError('Run not in release')
    r=matches[0]; seed=r['seed']; cell=r['cell'];config=r['config']
    # Output must never overlap frozen evidence or other runs.
    if a.out.exists(): raise FileExistsError('Use a new output directory')
    a.out.mkdir(parents=True)
    if a.method=='PS' and not (ROOT/'build/planning.so').exists():
        raise RuntimeError('First run: python scripts/build.py')
    learner=EmphasisLearner(cell,seed) if a.method=='PDR' else BaselineLearner(cell,seed,config)
    common=torch.load(ROOT/r['common_initial'],map_location='cpu',weights_only=False)
    statekeys=['models','optimizers','rng_torch','rng_cuda','rng_numpy','rng_python']
    base=learner.state() if a.method=='PDR' else learner.base_initial
    assert state_equal(base,{k:common[k] for k in statekeys}), 'Constructed start differs from source checkpoint'
    schedule_hash=hashlib.sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(learner.schedule.items()))).hexdigest()
    assert schedule_hash==common['schedule_sha256']
    initial=torch.load(ROOT/r['initial'],map_location='cpu',weights_only=False)
    initial=initial.get('state',initial)
    if a.method=='PDR':initial={k:initial[k] for k in statekeys}
    elif 'recipe' not in initial:initial['recipe']=copy.deepcopy(config)
    learner.restore(initial)
    if a.mode=='resume':
        if not r['early']:raise ValueError('No EARLY checkpoint was retained for this historical run')
        z=torch.load(ROOT/r['early'],map_location='cpu',weights_only=False);learner.restore(z['state']);start=z['completed_updates']
    elif a.mode=='final':
        z=torch.load(ROOT/r['final'],map_location='cpu',weights_only=False);learner.restore(z['state']);start=z['completed_updates']
    else:start=0
    if not start<=a.updates<=512 and a.mode!='final':raise ValueError('Invalid update limit')
    recorded=pd.read_csv(ROOT/r['curve']); rows=[]
    def measure(u):
        metrics=learner.evaluate(); target=recorded[recorded['update']==u]
        errors={}
        if len(target):
            errors={k:abs(float(v)-float(target.iloc[0][k])) for k,v in metrics.items() if k in target.columns}
            assert max(errors.values(),default=0)<=a.tolerance,(r['cell']['id'],a.method,u,errors)
        rows.append(dict(update=u,**metrics,max_recorded_metric_error=max(errors.values(),default=0)))
    measure(start)
    if a.mode!='final':
        for u in range(start,a.updates):
            learner.step(u,config) if a.method=='PDR' else learner.step(u)
            if u+1 in learner.checkpoints or u+1==a.updates:measure(u+1)
        torch.save(dict(state=learner.state(),completed_updates=a.updates,cell=cell,seed=seed,config=config),a.out/'REPRODUCED.pt')
    pd.DataFrame(rows).to_csv(a.out/'LEARNING_CURVE.csv',index=False)
    report=dict(status='PASS',task=a.task,method=a.method,seed=seed,mode=a.mode,
        start=start,end=rows[-1]['update'],max_metric_error=max(row['max_recorded_metric_error'] for row in rows),
        schedule_sha256=schedule_hash,tolerance=a.tolerance,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        numerical_code='unchanged source algorithms; only import paths relocated')
    (a.out/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
