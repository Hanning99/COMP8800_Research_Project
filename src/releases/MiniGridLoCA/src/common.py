from pathlib import Path
import os, sys, json, hashlib, time, ctypes, subprocess, random
ROOT=Path(__file__).resolve().parents[1]
if os.environ.get('BENCHMARK_CPU_AFFINITY'):
    os.sched_setaffinity(0,{int(i) for i in os.environ['BENCHMARK_CPU_AFFINITY'].split(',')})
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'vendor'))
os.environ['MINIGRID_LOCA_PROTOCOL']=str(ROOT/'inputs/loca_exact/PROTOCOL_LOCK.json')
os.environ['DEEPSEA_MAIN_UPDATES']='8192'
os.environ['DEEPSEA_EARLY_WINDOW']='2048'
import numpy as np
import torch
import pandas as pd
from threadpoolctl import threadpool_limits
torch.set_num_threads(1);torch.set_num_interop_threads(1)
BLAS_LIMIT=threadpool_limits(limits=1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(p)
def sync():torch.cuda.synchronize()
def ptr(x):return x.ctypes.data
def kernel():
    lib=ctypes.CDLL(str(ROOT/'build/kernels.so'))
    lib.deep_solve.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*6;lib.deep_solve.restype=None
    return lib
def seed_all(seed):
    torch.manual_seed(seed);np.random.seed(seed%2**32);random.seed(seed);torch.cuda.manual_seed_all(seed)
def flat_gradient(loss,actor):return torch.cat([x.detach().flatten() for x in torch.autograd.grad(loss,tuple(actor.parameters()))])

def state_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    return a==b

def assert_optimizer_steps(current,initial,updates):
    for oi,(o,base) in enumerate(zip(current,initial)):
        assert len(o['state'])>0,('Empty final optimizer',oi)
        for pi,s in o['state'].items():
            before=float(base['state'].get(pi,{}).get('step',0))
            assert float(s['step'])-before==updates,('Optimizer step mismatch',oi,pi,before,float(s['step']),updates)
