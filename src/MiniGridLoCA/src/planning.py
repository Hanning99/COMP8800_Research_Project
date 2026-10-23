from common import *

class Planner:
    def __init__(self,l):
        self.l=l;self.base=l.qold
        g=l.g if l.family=='MiniGridLoCA' else l.graph
        if l.family=='KeyCorridor':
            self.nxt=np.ascontiguousarray(g['next'],np.int64)
            self.terminal=np.ascontiguousarray(g['terminal'],np.uint8)
            self.valid=np.ascontiguousarray(g['mask'],np.uint8)
            self.reward=np.zeros_like(self.nxt,dtype=np.float64)
            self.depth=np.ascontiguousarray(g['min_depth'],np.int64)
            self.H=l.h;self.kind=1
        elif l.family=='MiniGridLoCA':
            self.nxt=np.ascontiguousarray(g.successor,np.int64)
            self.terminal=np.ascontiguousarray(g.done,np.uint8)
            self.valid=np.zeros_like(self.nxt,dtype=np.uint8);self.valid[g.nonterminal_states]=1
            self.reward=np.ascontiguousarray(g.reward_b,np.float64)
            self.depth=np.zeros(g.n_states,dtype=np.int64);self.H=0;self.kind=0
        else:
            self.nxt=np.ascontiguousarray(g['next'],np.int64)
            self.terminal=np.ascontiguousarray(g['terminal'],np.uint8)
            self.valid=np.ones_like(self.nxt,dtype=np.uint8)
            self.reward=np.ascontiguousarray(g['reward'],np.float64)
            self.depth=np.zeros(len(self.nxt),dtype=np.int64);self.H=0;self.kind=0
        self.S,self.A=self.nxt.shape
        assert self.base.flags.c_contiguous and self.base.dtype.itemsize in (4,8)
        self.lib=ctypes.CDLL(str(ROOT/'build/planning.so'))
        P=ctypes.c_void_p;I=ctypes.c_int;L=ctypes.c_int64
        self.lib.ps_new.argtypes=[I,I,I,I,ctypes.c_double,P,I,P,P,P,P,P];self.lib.ps_new.restype=P
        self.lib.ps_free.argtypes=[P]
        self.lib.ps_run.argtypes=[P,P,I,I]
        self.lib.ps_get.argtypes=[P,P,I,P]
        self.lib.ps_sizes.argtypes=[P,P]
        self.lib.ps_export.argtypes=[P,P,P,P,P]
        self.lib.ps_restore.argtypes=[P,P,P,L,P,P,L,L,L]
        self.handle=self.lib.ps_new(self.S,self.A,self.H,self.kind,l.gamma,ptr(self.base),self.base.dtype.itemsize,
            ptr(self.nxt),ptr(self.terminal),ptr(self.valid),ptr(self.reward),ptr(self.depth))

    def keys(self,b,all_actions=False):
        s=self.l.state_ids(b)
        t=b['t'].cpu().numpy() if self.kind==1 else np.zeros(len(s),np.int64)
        base=(t*self.S+s)*self.A
        if all_actions:return np.ascontiguousarray(base[:,None]+np.arange(self.A),np.int64)
        return np.ascontiguousarray(base+b['a'].cpu().numpy(),np.int64)

    def sizes(self):
        x=np.empty(4,np.int64);self.lib.ps_sizes(self.handle,ptr(x));return x

    def step(self,b,budget):
        keys=self.keys(b);before=self.sizes();self.lib.ps_run(self.handle,ptr(keys),len(keys),budget);after=self.sizes()
        return dict(backups=int(after[3]-before[3]),transition_reads=int(after[2]-before[2]),planning_entries=int(after[0]))

    def values(self,b):
        keys=self.keys(b,True);out=np.empty(keys.shape,np.float64)
        self.lib.ps_get(self.handle,ptr(keys),keys.size,ptr(out));return out

    def state(self):
        nv,np_,reads,backups=self.sizes()
        vk=np.empty(nv,np.int64);vv=np.empty(nv,np.float64);pk=np.empty(np_,np.int64);pv=np.empty(np_,np.float64)
        self.lib.ps_export(self.handle,ptr(vk),ptr(vv),ptr(pk),ptr(pv))
        vi=np.argsort(vk);pi=np.argsort(pk)
        return dict(value_keys=vk[vi],value_values=vv[vi],priority_keys=pk[pi],priority_values=pv[pi],reads=int(reads),backups=int(backups))

    def restore(self,z):
        self.lib.ps_restore(self.handle,ptr(z['value_keys']),ptr(z['value_values']),len(z['value_keys']),
            ptr(z['priority_keys']),ptr(z['priority_values']),len(z['priority_keys']),z['reads'],z['backups'])

    def __del__(self):
        if getattr(self,'handle',None):self.lib.ps_free(self.handle);self.handle=None
