"""Priorities relative to the original demo / uniform-state sampling law.

The full time/action domain is implicit. A sparse segment tree stores changes
to its initial probability mass, avoiding an 8.5-million-record replay copy.
"""
import heapq
import math
import numpy as np


class Domain:
    def __init__(self, graph, horizon, demos=None):
        self.demos = None if demos is None else np.asarray(demos, dtype=np.int64)[:, :3]
        if self.demos is not None:
            self.size = len(self.demos)
        else:
            self.depth=np.minimum(graph['min_depth'], horizon-1).astype(np.int64)
            self.actions=[np.flatnonzero(row) for row in graph['mask']]
            self.counts=np.asarray([len(x) for x in self.actions])
            assert (self.counts>0).all()
            self.lengths=(horizon-self.depth)*self.counts
            self.offsets=np.r_[0,np.cumsum(self.lengths)]
            self.size=int(self.offsets[-1]);self.states=len(self.lengths)

    def cdf(self, end):
        if end<=0:return 0.
        if end>=self.size:return 1.
        if self.demos is not None:return end/self.size
        s=int(np.searchsorted(self.offsets,end,side='right')-1)
        return (s+(end-self.offsets[s])/self.lengths[s])/self.states

    def probability(self, index):
        if self.demos is not None:return 1/self.size
        s=int(np.searchsorted(self.offsets,index,side='right')-1)
        return 1/(self.states*self.lengths[s])

    def decode(self, indices):
        indices=np.asarray(indices,dtype=np.int64)
        if self.demos is not None:return self.demos[indices].copy()
        s=np.searchsorted(self.offsets,indices,side='right')-1
        local=indices-self.offsets[s]
        t=self.depth[s]+local//self.counts[s]
        a=np.asarray([self.actions[i][j] for i,j in zip(s,local%self.counts[s])])
        return np.column_stack([s,t,a])

    def uniform(self, count, rng):
        if self.demos is not None:return self.decode(rng.integers(self.size,size=count))
        s=rng.integers(self.states,size=count)
        local=np.asarray([rng.integers(self.lengths[i]) for i in s])
        return self.decode(self.offsets[s]+local)


class Priorities:
    def __init__(self, domain, alpha):
        if not 0<=alpha<=1:raise ValueError(alpha)
        self.domain,self.alpha=domain,alpha
        self.width=1 << (domain.size-1).bit_length()
        self.delta={};self.factor={};self.minimum=[]

    @property
    def total(self):return 1+self.delta.get(1,0.)

    def sample(self, count, rng):
        out=[];ratios=[]
        for _ in range(count):
            mass=float(rng.random())*self.total
            node,left,right=1,0,self.width
            while right-left>1:
                middle=(left+right)//2
                weight=(self.domain.cdf(middle)-self.domain.cdf(left)
                        +self.delta.get(node*2,0.))
                if mass<weight:node,right=node*2,middle
                else:mass-=weight;node,left=node*2+1,middle
            if left>=self.domain.size:raise RuntimeError('priority mass outside domain')
            out.append(left);ratios.append(self.total/self.factor.get(left,1.))
        return np.asarray(out),np.asarray(ratios)

    def update(self, indices, errors):
        unique={}
        for i,e in zip(indices,errors):
            if not math.isfinite(float(e)):raise RuntimeError('nonfinite priority')
            i=int(i);unique[i]=max(unique.get(i,0.),abs(float(e))+1e-6)
        for i,p in sorted(unique.items()):
            factor=p**self.alpha
            change=self.domain.probability(i)*(factor-self.factor.get(i,1.))
            self.factor[i]=factor;heapq.heappush(self.minimum,(factor,i))
            node=self.width+i
            while node:
                self.delta[node]=self.delta.get(node,0.)+change;node//=2
        if self.total<=0:raise RuntimeError('nonpositive priority mass')

    def max_ratio(self):
        while self.minimum and self.factor[self.minimum[0][1]]!=self.minimum[0][0]:
            heapq.heappop(self.minimum)
        minimum=self.minimum[0][0] if self.minimum else 1.
        if len(self.factor)<self.domain.size:minimum=min(minimum,1.)
        return self.total/minimum

    def state_dict(self):
        return {'delta':self.delta,'factor':self.factor,'minimum':self.minimum}

    def load_state_dict(self,state):
        self.delta,self.factor,self.minimum=state['delta'],state['factor'],state['minimum']


def draw_batch(domains,rng):
    return np.concatenate([domains[0].uniform(58,rng),domains[1].uniform(6,rng)])


def draw_prioritized(pools,rng,beta):
    draws=[p.sample(n,rng) for p,n in zip(pools,[58,6])]
    normalizer=max(p.max_ratio() for p in pools)**beta
    weights=np.concatenate([r**beta/normalizer for _,r in draws])
    rows=np.concatenate([p.domain.decode(i) for p,(i,_) in zip(pools,draws)])
    return rows,weights,[i for i,_ in draws]
