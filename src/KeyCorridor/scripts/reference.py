"""Check, or rematerialize, neural reference values from the supplied critic."""
from pathlib import Path
import argparse,json,os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['S3R2','S3R3','S4R3','S5R3'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--materialize',action='store_true',help='Evaluate all state/time pairs and save a new table (never overwrite archived cache)')
    a=p.parse_args()
    from common import torch,np
    import key_exact as ref
    if not torch.cuda.is_available():raise RuntimeError('CUDA required to match the archived reference interface')
    if a.out.exists():raise FileExistsError('Use a new output directory')
    a.out.mkdir(parents=True)
    anchor=ROOT/'inputs/key_neural/anchors/confirmatory_anchor'/a.task/'AWAC'
    g=ref.load_graph(a.task);h=ref.TASKS[a.task][1]
    net=ref.base.Critic(g['features'].shape[1]).cuda()
    z=torch.load(anchor/'head_0/ANCHOR.pt',map_location='cpu',weights_only=False)
    net.load_state_dict(z['critic']);net.eval()
    x=torch.as_tensor(g['features'],device='cuda')
    expected=np.load(anchor/'NEURAL_QOLD.npy',mmap_mode='r')
    assert expected.shape==(h,len(x),7)
    if a.materialize:
        output=np.lib.format.open_memmap(a.out/'NEURAL_QOLD.npy',mode='w+',dtype=np.float32,shape=expected.shape)
        times=range(h)
    else:times=sorted(set([0,h//4,h//2,3*h//4,h-1]))
    error=0.;count=0
    with torch.no_grad():
        for t in times:
            for i in range(0,len(x),8192):
                b=x[i:i+8192];tf=torch.full((len(b),),t/h,dtype=torch.float32,device='cuda')
                y=net(b,tf).cpu().numpy();error=max(error,float(np.max(np.abs(y-expected[t,i:i+len(b)]))))
                if a.materialize:output[t,i:i+len(b)]=y
                count+=y.size
    if a.materialize:output.flush()
    assert error<1e-5,error
    report=dict(status='PASS',task=a.task,values_checked=count,all_time_layers=a.materialize,
        max_neural_cache_error=error,scope='Neural checkpoint/cache consistency, NOT accuracy against true policy values')
    (a.out/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
