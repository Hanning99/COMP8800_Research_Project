"""Discrete IQL, discrete SAC-CQL, and matched replay/planning baselines.

The environment, replay, original model classes and evaluators are inherited
from the verified public continuations. IQL/CQL have their own actor/critic
objectives; they are not renamed AWAC weighting interventions.
"""
from learner import *
import torch.nn as nn
from priority_sampling import Domain, Priorities


class UniformDomain:
    def __init__(self, size): self.size = int(size)
    def cdf(self, end): return min(max(end, 0), self.size) / self.size
    def probability(self, index): return 1. / self.size


def capped_exp(advantage, inverse_temperature, cap):
    # Clamp before exp to avoid overflowing intermediates; same capped function.
    return (advantage * inverse_temperature).clamp(max=float(np.log(cap))).exp().detach()


def expectile_loss(diff, tau):
    return (torch.where(diff > 0, tau, 1-tau) * diff.square()).mean()


def conservative_penalty(q, action, valid):
    return (q.masked_fill(~valid, -torch.inf).logsumexp(1)
            - q.gather(1, action[:, None]).squeeze(1)).mean()


class BaselineLearner(Learner):
    def __init__(self, cell, seed, recipe):
        super().__init__(cell, seed)
        self.recipe = recipe
        self.method = recipe['method']
        self.base_model_count = len(self.models)
        self.base_optimizer_count = len(self.optimizers)
        self.base_initial = super().state()
        self.rng = np.random.default_rng(seed + 20260926)
        self.counts = dict(actor=0, critic=0, value=0, target=0, error=0, planning=0)
        self.qmods = self.critics if self.family == 'MiniGridLoCA' else [self.critic]
        self.qtargets = self.targets if self.family == 'MiniGridLoCA' else [self.target]
        self.qopts = self.opts if self.family == 'MiniGridLoCA' else [self.co]
        self.gamma = .97 if self.family == 'KeyCorridor' else (.99 if self.family == 'MiniGridLoCA' else self.core.GAMMA)
        self.tau = self.study.TAU if self.family == 'MiniGridLoCA' else .005
        self.per = None; self.planner = None; self.error = None
        self.initial_policy = self.policy()
        if self.method in ['IQL', 'CQL']:
            # Warm-start every head from the available original critic. A second
            # available head is retained; otherwise the initial head is cloned.
            if len(self.qmods) == 1:
                q2 = copy.deepcopy(self.qmods[0]); t2 = copy.deepcopy(self.qtargets[0])
                o2 = torch.optim.Adam(q2.parameters(), lr=self.qopts[0].param_groups[0]['lr'])
                o2.load_state_dict(copy.deepcopy(self.qopts[0].state_dict()))
                self.qmods = self.qmods + [q2]; self.qtargets = self.qtargets + [t2]; self.qopts = self.qopts + [o2]
                self.models.extend([q2,t2]); self.optimizers.append(o2)
        if self.method == 'IQL':
            self.anchors = [copy.deepcopy(m).requires_grad_(False) for m in self.qmods]
            self.models.extend(self.anchors)
            self.value_model = copy.deepcopy(self.qmods[0])
            if hasattr(self.value_model, 'head'):
                original = self.value_model.head
                self.value_model.head = nn.Linear(original.in_features,1).to(self.device)
                head = self.value_model.head
            else:
                original = self.value_model.net[-1]
                self.value_model.net[-1] = nn.Linear(original.in_features,1).to(self.device)
                head = self.value_model.net[-1]
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
            self.vo = torch.optim.Adam(self.value_model.parameters(),lr=self.qopts[0].param_groups[0]['lr'])
            self.models.append(self.value_model); self.optimizers.append(self.vo)
        if self.method in ['PER', 'EXTRA_TD']:
            self.prepare_domains()
        if self.method == 'PER':
            self.per = [Priorities(d, .4) for d in self.domains]
        if self.method == 'DISCOR':
            self.error = copy.deepcopy(self.qmods[0])
            if hasattr(self.error,'head'):
                n=self.error.head.in_features; a=self.error.head.out_features
                self.error.head=nn.Sequential(nn.Linear(n,n),nn.LeakyReLU(.01),nn.Linear(n,a),nn.Softplus()).to(self.device)
            else:
                n=self.error.net[-1].in_features; a=self.error.net[-1].out_features
                self.error.net=nn.Sequential(*list(self.error.net.children())[:-1],nn.Linear(n,n),nn.ReLU(),nn.Linear(n,a),nn.Softplus()).to(self.device)
            self.error_target=copy.deepcopy(self.error).requires_grad_(False)
            self.eo=torch.optim.Adam(self.error.parameters(),lr=1e-4)
            self.error_temperature=10.
            self.models.extend([self.error,self.error_target]);self.optimizers.append(self.eo)
        if self.method == 'PS':
            from planning import Planner
            self.planner=Planner(self)

    def pair(self, u):
        b = self.batch(u)
        if self.family != 'MiniGridLoCA': return b,b
        def convert(s,a):
            ns=self.nxt_t[s,a]
            return dict(s=s, a=a, x=s, nx=ns, ns=ns,
                        mask=torch.ones((len(s),3),dtype=torch.bool,device=self.device),
                        nmask=torch.ones((len(s),3),dtype=torch.bool,device=self.device),
                        reward=self.reward_t[s,a],done=self.done_t[s,a])
        return convert(b['cs'],b['ca']),convert(b['s'],b['a'])

    def forward(self, module, b, successor=False):
        x=b['nx'] if successor else b['x']
        if self.family == 'KeyCorridor':return module(x,b['ntf'] if successor else b['tf'])
        return module(x)

    def q(self, module, b, successor=False):
        value=self.forward(module,b,successor)
        if self.family=='KeyCorridor' and self.cell['reference']=='exact':
            value=value+(b['qold_next'] if successor else b['qold'])
        return value

    def valid_mask(self,b,successor=False):
        key='nmask' if successor else 'mask'
        if key in b:return b[key]
        return torch.ones((len(b['a']),3 if self.family=='MiniGridLoCA' else 2),dtype=torch.bool,device=self.device)

    def logprob(self,b,successor=False):
        x=b['nx'] if successor else b['x']
        if self.family=='KeyCorridor':return self.actor(x,self.valid_mask(b,successor))
        if self.family=='MiniGridLoCA':return self.actor.log_policy(x)
        return self.actor(x)

    def reward(self,b):return b['r'] if self.family=='DeepSea' else b['reward']

    def state_ids(self,b,successor=False):
        if self.family=='DeepSea':
            states=np.asarray(b['s_id'])
            if successor:states=np.maximum(0,self.graph['next'][states,b['a_np']])
            return states
        return (b['ns'] if successor else b['s']).detach().cpu().numpy()

    def v(self,b,successor=False):
        # The additional value model starts at the old actor's expectation of
        # the common initial neural critic. Its residual is initially zero.
        with torch.no_grad():
            p=torch.as_tensor(self.initial_policy[self.state_ids(b,successor)],dtype=torch.float32,device=self.device)
            q=torch.stack([self.q(m,b,successor) for m in self.anchors]).min(0).values
            anchor=(p*q).sum(1)
        return anchor+self.forward(self.value_model,b,successor).squeeze(1)

    def optimize(self,loss,model,optimizer,clip=True):
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if clip and self.clip is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),self.clip)
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():raise RuntimeError('Nonfinite gradient')
        optimizer.step()

    def target_step(self):
        with torch.no_grad():
            for t,q in zip(self.qtargets,self.qmods):
                for a,b in zip(t.parameters(),q.parameters()):a.mul_(1-self.tau).add_(b,alpha=self.tau)
        self.counts['target']+=1

    def native_actor(self,b,values):
        # Reuse the complete-weight detach convention of the verified runs.
        return Learner.objective(self,b,values)

    def native_target(self,b):
        with torch.no_grad():
            lp=self.logprob(b,True);p=lp.exp()
            value=torch.stack([self.q(m,b,True) for m in self.qtargets]).min(0).values
            value=value-(getattr(self,'alpha',0.)*lp if self.backbone=='SAC' else 0.)
            return self.reward(b)+self.gamma*(1-b['done'])*(p*value).sum(1),p

    def regress(self,b,target,weights=None):
        errors=[];losses=[]
        for m,opt in zip(self.qmods,self.qopts):
            pred=self.q(m,b).gather(1,b['a'][:,None]).squeeze(1)
            error=pred-target;errors.append(error.detach())
            loss=(error.square()*(1. if weights is None else weights.detach())).mean()
            self.optimize(loss,m,opt);losses.append(float(loss.detach()))
        self.counts['critic']+=1
        return torch.stack(errors).abs().max(0).values.cpu().numpy(),float(np.mean(losses))

    def prepare_domains(self):
        if self.family=='KeyCorridor':
            self.domains=[Domain(self.graph,self.h,self.ref.demonstration(self.graph)),Domain(self.graph,self.h)]
            self.pool_counts=[58,6]
        elif self.family=='MiniGridLoCA':
            self.pool_states=np.repeat(self.g.nonterminal_states,self.g.n_actions)
            self.pool_actions=np.tile(np.arange(self.g.n_actions),len(self.g.nonterminal_states))
            self.domains=[UniformDomain(len(self.pool_states))];self.pool_counts=[self.study.BATCH]
        else:
            self.domains=[UniformDomain(len(self.hist)),UniformDomain(len(self.support))];self.pool_counts=[16,48]

    def sampled_batch(self, prioritized=False, u=0):
        if prioritized:
            draws=[p.sample(n,self.rng) for p,n in zip(self.per,self.pool_counts)]
            beta=.4+.6*u/max(1,self.updates-1)
            norm=max(p.max_ratio() for p in self.per)**beta
            weights=np.concatenate([r**beta/norm for _,r in draws]);indices=[i for i,_ in draws]
        else:
            indices=[self.rng.integers(d.size,size=n) if isinstance(d,UniformDomain) else None for d,n in zip(self.domains,self.pool_counts)]
            weights=np.ones(sum(self.pool_counts))
        if self.family=='KeyCorridor':
            rows=np.concatenate([d.decode(i) if i is not None else d.uniform(n,self.rng) for d,i,n in zip(self.domains,indices,self.pool_counts)])
            b=self.ref.batch_tensors(self.graph,self.h,self.qold,*rows.T)
        elif self.family=='MiniGridLoCA':
            ids=indices[0];s=torch.as_tensor(self.pool_states[ids],device=self.device);a=torch.as_tensor(self.pool_actions[ids],device=self.device)
            ns=self.nxt_t[s,a]
            b=dict(s=s,a=a,x=s,nx=ns,ns=ns,reward=self.reward_t[s,a],done=self.done_t[s,a])
        else:
            ids=np.concatenate(indices);sources=np.repeat([0,1],self.pool_counts)
            order=self.rng.permutation(len(ids));ids=ids[order];sources=sources[order];weights=weights[order]
            b=self.core.rows_to_batch(self.hist,self.support,ids,sources,self.graph)
            b['priority_order']=order
        return b,torch.as_tensor(weights,dtype=torch.float32,device=self.device),indices

    def step(self,u,arm=None):
        if self.method in ['TD','OURS']:
            raw=self.batch(u);Learner.critic_step(self,raw)
            if self.method=='OURS':
                old,delta,corrected=self.values(raw);alpha,beta=self.recipe['coefficients']
                values=beta*corrected if alpha==beta else (beta*delta if alpha==0 else alpha*old+beta*delta)
            else:values=self.td_values(raw)*self.recipe['gain']
            result=Learner.actor_step(self,raw,values)
            self.counts['actor']+=1;self.counts['critic']+=1;self.counts['target']+=1
            self.last_info=dict(actor_loss=float(result[0].detach()),signal_rms=float(result[1].square().mean().sqrt()))
            return result
        cb,ab=self.pair(u);method=self.method;info={}
        if method=='IQL':
            with torch.no_grad(): q=torch.stack([self.q(t,cb) for t in self.qtargets]).min(0).values.gather(1,cb['a'][:,None]).squeeze(1)
            loss_v=expectile_loss(q-self.v(cb),self.recipe['expectile'])
            self.optimize(loss_v,self.value_model,self.vo);self.counts['value']+=1
            with torch.no_grad():
                q=torch.stack([self.q(t,ab) for t in self.qtargets]).min(0).values
                signal=q-self.v(ab)[:,None]
                weights=capped_exp(signal,self.recipe['inverse_temperature'],100.)
            lp=self.logprob(ab);actor_loss=-(weights.gather(1,ab['a'][:,None])*lp.gather(1,ab['a'][:,None])).mean()
            self.optimize(actor_loss,self.actor,self.ao)
            with torch.no_grad():target=self.reward(cb)+self.gamma*(1-cb['done'])*self.v(cb,True)
            _,ql=self.regress(cb,target);info.update(value_loss=float(loss_v.detach()),critic_loss=ql)
        elif method=='CQL':
            entropy=self.recipe['entropy_temperature']
            with torch.no_grad():
                nlp=self.logprob(cb,True)
                nq=torch.stack([self.q(t,cb,True) for t in self.qtargets]).min(0).values
                target=self.reward(cb)+self.gamma*(1-cb['done'])*(nlp.exp()*(nq-entropy*nlp)).sum(1)
            penalties=[]
            for m,opt in zip(self.qmods,self.qopts):
                q=self.q(m,cb);td=F.mse_loss(q.gather(1,cb['a'][:,None]).squeeze(1),target)
                penalty=conservative_penalty(q,cb['a'],self.valid_mask(cb));penalties.append(float(penalty.detach()))
                self.optimize(td+self.recipe['conservatism']*penalty,m,opt)
            self.counts['critic']+=1
            with torch.no_grad():q=torch.stack([self.q(m,ab) for m in self.qmods]).min(0).values
            lp=self.logprob(ab);actor_loss=(lp.exp()*(entropy*lp-q)).sum(1).mean()
            signal=q-(lp.exp().detach()*q).sum(1,keepdim=True);weights=None
            self.optimize(actor_loss,self.actor,self.ao);info['conservative_penalty']=float(np.mean(penalties))
        else:
            if method=='PS':
                reads=self.planner.step(cb,int(self.recipe['backups']))
                self.counts['planning']+=reads['backups'];info.update(reads)
                values=torch.as_tensor(self.planner.values(ab),dtype=torch.float32,device=self.device)*self.recipe['gain']
            else:
                ratio=8 if method=='EXTRA_TD' else 1
                for j in range(ratio):
                    b=cb if j==0 else self.sampled_batch()[0]
                    weights_critic=None;indices=None
                    if method=='PER':b,weights_critic,indices=self.sampled_batch(True,u)
                    target,policy=self.native_target(b)
                    if method=='DISCOR':
                        with torch.no_grad():
                            future=self.gamma*(1-b['done'])*(policy*self.forward(self.error_target,b,True)).sum(1)
                            pred=torch.stack([self.q(m,b) for m in self.qmods]).min(0).values.gather(1,b['a'][:,None]).squeeze(1)
                            error_target=(pred-target).abs()+future
                            weights_critic=torch.softmax(-future/max(self.error_temperature,1e-8),dim=0)*len(future)
                        estimate=self.forward(self.error,b).gather(1,b['a'][:,None]).squeeze(1)
                        error_mean=float(estimate.detach().mean())
                        error_loss=F.mse_loss(estimate,error_target)
                    errors,ql=self.regress(b,target,weights_critic)
                    if method=='PER':
                        if 'priority_order' in b:
                            unsorted=np.empty_like(errors);unsorted[b['priority_order']]=errors;errors=unsorted
                        offset=0
                        for pool,ids,n in zip(self.per,indices,self.pool_counts):pool.update(ids,errors[offset:offset+n]);offset+=n
                    if method=='DISCOR':
                        self.optimize(error_loss,self.error,self.eo);self.counts['error']+=1
                        with torch.no_grad():
                            for a,z in zip(self.error_target.parameters(),self.error.parameters()):a.mul_(.995).add_(z,alpha=.005)
                        self.error_temperature=.995*self.error_temperature+.005*error_mean
                    if j+1<ratio:self.target_step()
                if method=='OURS':
                    old,delta,corrected=self.values(ab)
                    alpha,beta=self.recipe['coefficients']
                    # Preserve arithmetic of each already-selected branch.
                    values=beta*corrected if alpha==beta else (beta*delta if alpha==0 else alpha*old+beta*delta)
                else:values=self.td_values(ab)*self.recipe['gain']
            actor_loss,signal,weights=self.native_actor(ab,values)
            self.optimize(actor_loss,self.actor,self.ao)
        self.counts['actor']+=1
        if method!='PS':self.target_step()
        assert torch.isfinite(signal).all()
        info.update(actor_loss=float(actor_loss.detach()),signal_rms=float(signal.square().mean().sqrt()))
        if weights is not None:info['weight_mean']=float(weights.gather(1,ab['a'][:,None]).mean())
        self.last_info=info
        return actor_loss,signal,weights

    def state(self):
        z=super().state()
        z.update(recipe=copy.deepcopy(self.recipe),counts=dict(self.counts),sampling_rng=copy.deepcopy(self.rng.bit_generator.state),
                 error_temperature=getattr(self,'error_temperature',None),
                 priorities=copy.deepcopy([p.state_dict() for p in self.per]) if self.per is not None else None,
                 planning=self.planner.state() if self.planner is not None else None)
        return z

    def restore(self,z):
        assert z['recipe']==self.recipe
        super().restore(z)
        self.counts=dict(z['counts']);self.rng.bit_generator.state=copy.deepcopy(z['sampling_rng'])
        if self.error is not None:self.error_temperature=z['error_temperature']
        if self.per is not None:
            for p,s in zip(self.per,z['priorities']):p.load_state_dict(copy.deepcopy(s))
        if self.planner is not None:self.planner.restore(z['planning'])
