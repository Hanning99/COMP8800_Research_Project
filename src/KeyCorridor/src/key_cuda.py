"""Capture full finite-horizon PDR recurrences, reusing buffers, never values.

The graph records operator launches. Current policy and current batch indices
are copied to its input buffers at every refresh. All horizon layers are kept.
"""
import torch

class KeyCUDASolve:
    def __init__(self,oldq,oldpi,graph,batch_size=64,gamma=.97):
        self.oldq=oldq; self.oldpi=oldpi;self.gamma=gamma
        device=oldq.device;self.h,self.n,self.a=oldq.shape
        self.pi=oldpi.clone();self.diff=torch.empty_like(oldpi)
        self.next=torch.as_tensor(graph['next'].clip(0).reshape(-1),dtype=torch.long,device=device)
        self.live=torch.as_tensor(graph['mask'] & ~graph['terminal'],dtype=torch.float32,device=device)
        self.products=torch.empty_like(oldpi);self.d=torch.empty_like(oldpi)
        self.source=torch.empty(self.n,device=device);self.v=torch.zeros_like(self.source);self.total=torch.empty_like(self.source)
        self.bs=torch.zeros(batch_size,dtype=torch.long,device=device);self.bt=torch.zeros_like(self.bs)
        self.selected=torch.empty((batch_size,self.a),device=device)
        self.contribution=torch.empty_like(self.selected);self.output=torch.empty_like(self.selected)
        self.active=torch.empty(batch_size,dtype=torch.bool,device=device)
        self.graph=torch.cuda.CUDAGraph()
        # Initialize libraries and kernels on a side stream before capture.
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream),torch.no_grad():self._operations()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        with torch.no_grad(),torch.cuda.graph(self.graph):self._operations()
        torch.cuda.synchronize(device)
    def _operations(self):
        torch.sub(self.pi,self.oldpi,out=self.diff)
        self.v.zero_();self.output.zero_()
        # D at horizon-1 is zero; all earlier contributions are included.
        for t in range(self.h-2,-1,-1):
            torch.mul(self.diff,self.oldq[t+1],out=self.products)
            torch.sum(self.products,dim=1,out=self.source)
            torch.add(self.source,self.v,out=self.total)
            torch.index_select(self.total,0,self.next,out=self.d.view(-1))
            self.d.mul_(self.gamma).mul_(self.live)
            torch.index_select(self.d,0,self.bs,out=self.selected)
            torch.eq(self.bt,t,out=self.active)
            torch.mul(self.selected,self.active[:,None],out=self.contribution)
            self.output.add_(self.contribution)
            torch.mul(self.pi,self.d,out=self.products)
            torch.sum(self.products,dim=1,out=self.v)
    def __call__(self,policy,batch):
        self.pi.copy_(policy);self.bs.copy_(batch['s']);self.bt.copy_(batch['t'])
        self.graph.replay()
        return batch['qold']+self.output
