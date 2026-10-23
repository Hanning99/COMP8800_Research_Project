"""Cross actor objective and value source with matched SAC-CQL critic updates."""
from baselines import *

class ContrastLearner(BaselineLearner):
 def __init__(self,cell,seed,recipe,condition):
  super().__init__(cell,seed,recipe);assert self.family=='DeepSea'
  self.condition=condition;self.entropy=self.recipe['entropy_temperature']
  reference_pi=np.asarray(self.oldpi,dtype=np.float64)
  self.soft_old=self.direct_soft_q(reference_pi)
  self.old_aggregate=(reference_pi*(self.soft_old-self.entropy*np.log(np.clip(reference_pi,1e-30,1)))).sum(1)

 def direct_soft_q(self,pi):
  pi=np.asarray(pi,dtype=np.float64)
  g=self.graph;q=np.zeros_like(self.qold,dtype=np.float64);v=np.zeros(len(pi),np.float64)
  for layer in range(int(g['size'])-1,-1,-1):
   ids=np.flatnonzero(g['states'][:,0]==layer);q[ids]=g['reward'][ids]
   if layer<int(g['size'])-1:q[ids]+=v[g['next'][ids]]
   v[ids]=(pi[ids]*(q[ids]-self.entropy*np.log(np.clip(pi[ids],1e-30,1)))).sum(1)
  return q

 @torch.no_grad()
 def corrected(self):
  pi=np.ascontiguousarray(self.policy(),np.float64)
  source=np.ascontiguousarray((pi*(self.soft_old-self.entropy*np.log(np.clip(pi,1e-30,1)))).sum(1)-self.old_aggregate)
  q=np.empty_like(self.soft_old)
  self.lib.deep_solve(len(pi),ptr(self.nxt),ptr(self.live),ptr(pi),ptr(source),ptr(self.soft_old),ptr(q))
  return q

 def step(self,u,arm=None):
  # The native control executes the archived baseline implementation directly.
  if self.condition['id']=='LIVE_SAC':return super().step(u)
  cb,ab=self.pair(u);entropy=self.entropy
  with torch.no_grad():
   nlp=self.logprob(cb,True);nq=torch.stack([self.q(t,cb,True) for t in self.qtargets]).min(0).values
   target=self.reward(cb)+self.gamma*(1-cb['done'])*(nlp.exp()*(nq-entropy*nlp)).sum(1)
  penalties=[]
  for m,opt in zip(self.qmods,self.qopts):
   q=self.q(m,cb);td=F.mse_loss(q.gather(1,cb['a'][:,None]).squeeze(1),target)
   penalty=conservative_penalty(q,cb['a'],self.valid_mask(cb));penalties.append(float(penalty.detach()))
   self.optimize(td+self.recipe['conservatism']*penalty,m,opt)
  self.counts['critic']+=1
  with torch.no_grad():
   if self.condition['source']=='LIVE':value=torch.stack([self.q(m,ab) for m in self.qmods]).min(0).values
   else:
    q=self.corrected();a=self.condition['alpha'];b=self.condition['beta']
    value=torch.as_tensor((a*self.soft_old+b*(q-self.soft_old))[ab['s_id']],dtype=torch.float32,device=self.device)
  lp=self.logprob(ab);signal=value-(lp.exp().detach()*value).sum(1,keepdim=True)
  if self.condition['actor']=='SAC':
   actor_loss=(lp.exp()*(entropy*lp-value)).sum(1).mean();weights=None
  else:
   weights=torch.exp(signal/self.beta).clamp(0.,self.cap).detach()
   actor_loss=-(weights.gather(1,ab['a'][:,None])*lp.gather(1,ab['a'][:,None])).mean()
  self.optimize(actor_loss,self.actor,self.ao);self.counts['actor']+=1;self.target_step()
  assert torch.isfinite(signal).all()
  self.last_info=dict(actor_loss=float(actor_loss.detach()),signal_rms=float(signal.square().mean().sqrt()),conservative_penalty=float(np.mean(penalties)))
  if weights is not None:self.last_info['weight_mean']=float(weights.gather(1,ab['a'][:,None]).mean())
  return actor_loss,signal,weights

 def diagnose(self):
  q=self.corrected();direct=self.direct_soft_q(self.policy());error=float(np.max(np.abs(q-direct)))
  assert error<1e-10,('Soft PDR disagrees with independent current-policy evaluation',error)
  return dict(soft_pdr_direct_max_abs=error,delta_max_abs=float(np.max(np.abs(q-self.soft_old))))
