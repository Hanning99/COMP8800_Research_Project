"""Portable replay/evaluation of one archived DeepSea continuation."""
from pathlib import Path
import argparse,os,sys,json,hashlib,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,choices=[10,20,30,40],required=True)
    p.add_argument('--method',choices=['TD','PER','DISCOR','EXTRA_TD','IQL','CQL','PS','PDR','OURS'],required=True)
    p.add_argument('--seed',type=int);p.add_argument('--updates',type=int,default=8192)
    p.add_argument('--mode',choices=['replay','final'],default='replay')
    p.add_argument('--resume',type=Path,help='Resume a checkpoint created by this script.')
    p.add_argument('--tolerance',type=float,default=1e-5);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert not a.out.exists(),'Output must be new'
    a.out.mkdir(parents=True)
    from contrast import ContrastLearner,BaselineLearner,torch,np,pd,state_equal
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose exactly one CUDA GPU'
    records=json.loads((R/'RUNS.json').read_text());method='OURS' if a.method=='PDR' else a.method
    matches=[r for r in records if r['N']==a.size and r['method']==method and (a.seed is None or r['seed']==a.seed)]
    assert matches,'No archived run matches';r=matches[0];seed=r['seed']
    l=(ContrastLearner(r['cell'],seed,r['recipe'],r['condition']) if r['condition'] else BaselineLearner(r['cell'],seed,r['recipe']))
    load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
    base=load(R/r['base_initial']);assert state_equal(l.base_initial,{k:base[k] for k in l.base_initial}),'Common start differs'
    sh=hashlib.sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(l.schedule.items()))).hexdigest()
    assert sh==r['schedule_sha256']
    reference=load(R/r['exact_reference']);referror=0.
    if r['condition']:
        assert np.array_equal(l.oldpi,reference['policy'])
        referror=float(np.max(np.abs(l.soft_old-reference['soft_qold'])))
        assert referror<1e-12,referror
        # Use the supplied exact reference table, without reference pretraining.
        l.soft_old=reference['soft_qold'].copy()
        pi=np.asarray(l.oldpi,np.float64)
        l.old_aggregate=(pi*(l.soft_old-l.entropy*np.log(np.clip(pi,1e-30,1)))).sum(1)
    initial=load(R/r['initial']);assert state_equal(l.state(),initial),'Method initialization differs'
    l.restore(initial)
    original=pd.read_csv(R/r['curve']).set_index('update')
    with np.load(R/Path(r['curve']).parent/'POLICIES.npz') as z:
        policies={int(u):pi for u,pi in zip(z['updates'],z['policy'])}
    if a.mode=='final':
        z=load(R/r['final']);l.restore(z['state']);start=end=8192
    elif a.resume:
        z=load(a.resume);assert z['run']=={k:r[k] for k in ['N','seed','method']}
        l.restore(z['state']);start=int(z['update']);end=a.updates
    else:start=0;end=a.updates
    assert 0<=start<=end<=8192
    rows=[];maxerror=0.;policyerror=0.;diagnostic=0.
    def measure(u):
        nonlocal maxerror,policyerror,diagnostic
        metrics=l.evaluate();rows.append(dict(update=u,**metrics))
        if u in original.index:
            for k,v in metrics.items():
                if k in original.columns:maxerror=max(maxerror,abs(float(v)-float(original.loc[u,k])))
            policyerror=max(policyerror,float(np.max(np.abs(l.policy()-policies[u]))))
            assert maxerror<=a.tolerance,(u,maxerror,'metric mismatch')
            assert policyerror<=a.tolerance,(u,policyerror,'policy mismatch')
        if r['condition']:diagnostic=max(diagnostic,l.diagnose()['soft_pdr_direct_max_abs'])
    t=time.perf_counter();measure(start)
    for u in range(start,end):
        l.step(u)
        if u+1 in l.checkpoints or u+1==end:measure(u+1)
    torch.cuda.synchronize()
    if a.mode!='final':
        torch.save(dict(state=l.state(),update=end,curves=rows,run={k:r[k] for k in ['N','seed','method']}),a.out/'REPRODUCED.pt')
    pd.DataFrame(rows).to_csv(a.out/'LEARNING_CURVE.csv',index=False)
    report=dict(status='PASS',N=a.size,method=method,seed=seed,mode=a.mode,start=start,end=end,
        max_metric_error=maxerror,max_policy_error=policyerror,reference_max_error=referror,
        pdr_direct_soft_value_max_error=diagnostic,tolerance=a.tolerance,
        common_start_bitwise_matched=True,method_start_bitwise_matched=True,
        schedule_sha256=sh,seconds=time.perf_counter()-t,torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
    (a.out/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
