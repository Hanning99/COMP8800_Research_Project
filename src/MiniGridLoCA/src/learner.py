"""Matched actor-signal intervention on archived benchmark continuations."""
from common import *
import copy
import torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class Learner:
    def __init__(self,cell,seed):
        self.cell=cell;self.seed=seed;self.family=cell['family'];self.backbone=cell['backbone'];self.device=torch.device(os.environ.get('MINIGRID_RELEASE_DEVICE','cuda:0'))
        seed_all(seed);self.inputs=[]
        if self.family=='KeyCorridor':self.init_key()
        elif self.family=='MiniGridLoCA':self.init_loca()
        else:self.init_deep()
        self.updates=cell['updates']

    def init_key(self):
        import key_exact as ref
        ref.DEVICE=self.device;ref.base.DEVICE=self.device;self.ref=ref
        self.task=self.cell['task'];self.graph=ref.load_graph(self.task);self.h=ref.TASKS[self.task][1]
        self.common,self.actor,self.ao,self.critic,self.target,self.co=ref.initialize_models(self.task,self.graph,self.seed)
        if self.cell['reference']=='neural':
            p=ROOT/'inputs/key_neural/anchors/confirmatory_anchor'/self.task/'AWAC'
            raw=torch.load(p/'head_0/ANCHOR.pt',map_location=self.device,weights_only=False)
            self.critic.load_state_dict(raw['critic']);self.target.load_state_dict(raw['critic'])
            self.co=torch.optim.Adam(self.critic.parameters(),lr=ref.LEARNING_RATE)
            self.qpath=p/'NEURAL_QOLD.npy';self.inputs.append(p/'head_0/ANCHOR.pt')
            schedule=ROOT/'inputs/key_neural/preregistration/schedules/confirmatory'/self.task/f'seed_{self.seed}.npz'
        else:
            self.qpath=ref.source_paths(self.task)['qold'];schedule=ref.schedule_path(self.task,self.seed)
        self.qold=np.load(self.qpath,mmap_mode='r');self.oldpi=self.common['pi_old'];self.schedule=dict(np.load(schedule))
        self.qdevice=torch.tensor(self.qold,device=self.device);self.oldpi_t=torch.tensor(self.oldpi,dtype=torch.float32,device=self.device)
        self.features=torch.as_tensor(self.graph['features'],device=self.device);self.mask=torch.as_tensor(self.graph['mask'],device=self.device)
        from key_cuda import KeyCUDASolve
        self.solver=KeyCUDASolve(self.qdevice,self.oldpi_t,self.graph)
        self.inputs += [self.qpath,schedule,ref.source_paths(self.task)['common'],ref.source_paths(self.task)['graph']]
        self.models=[self.actor,self.critic,self.target];self.optimizers=[self.ao,self.co]
        self.checkpoints=ref.CHECKPOINTS;self.beta=ref.BETA;self.cap=ref.WEIGHT_CLIP;self.clip=ref.GRAD_CLIP

    def init_loca(self):
        import run_study as study
        import minigrid_loca_core as core
        self.study=study;self.core=core;self.g=core.build_graph();g=self.g
        if self.cell['reference']=='exact':
            self.raw,self.actor,self.ao,self.critics,self.targets,self.opts=study.load_branch(self.backbone,self.seed,self.device,'confirmatory')
            path=study.ROOT/'branches'/study.ARTIFACT_VERSION/'confirmatory'/f'seed_{self.seed}'/self.backbone/'branch.pt'
            self.qold=np.asarray(self.raw['qold'],np.float64)
        else:
            import loca_neural as m
            path,self.raw,self.actor,self.ao,self.critics,self.targets,self.opts=m.load_branch(self.backbone,self.seed,'confirmatory',self.device)
            self.qold=m.actor_facing_table(self.critics,study.all_state_tensor(g,self.device),self.backbone,g)
        self.inputs=[path,study.CONFIG_PATH];self.oldpi=np.asarray(self.raw['pi_old'],np.float64)
        self.alpha=study.SAC_ALPHA if self.backbone=='SAC' else 0.
        self.schedule=study.schedules(g,self.seed);self.states=study.all_state_tensor(g,self.device)
        self.nxt_t=torch.as_tensor(g.successor,device=self.device);self.done_t=torch.as_tensor(g.done,dtype=torch.float32,device=self.device)
        self.reward_t=torch.as_tensor(g.reward_b,dtype=torch.float32,device=self.device)
        self.rows,self.actions=np.where(~g.done);keep=np.isin(self.rows,g.nonterminal_states)
        self.rows=self.rows[keep];self.actions=self.actions[keep];self.succ=g.successor[self.rows,self.actions]
        self.terminal=np.setdiff1d(np.arange(g.n_states),g.nonterminal_states)
        self.oldaggregate=(self.oldpi*(self.qold-self.alpha*np.log(np.clip(self.oldpi,1e-15,1)))).sum(1)
        self.reward_delta=g.reward_b.astype(np.float64)-g.reward_a.astype(np.float64);self.identity=sp.eye(g.n_states,format='csc')
        self.models=[self.actor,*self.critics,*self.targets];self.optimizers=[self.ao,*self.opts]
        self.checkpoints=study.CHECKPOINTS;self.beta=study.AWAC_BETA;self.cap=study.AWAC_CLIP;self.clip=None

    def init_deep(self):
        import deepsea_experiment as core
        core.DEVICE=self.device;core.ROOT=ROOT/'inputs/deepsea';self.core=core
        self.wdir=core.world_dir(self.cell['N'],self.seed)
        self.graph,self.common,self.actor,self.critic,self.target,self.ao,self.co=core.load_branch_models(self.cell['N'],self.seed)
        self.hist,self.support=core.load_replay(self.wdir);self.schedule=dict(np.load(self.wdir/'schedule.npz'))
        self.offset=int(self.common['anchor_update']);self.oldpi=self.common['pi0'];self.qold=np.ascontiguousarray(self.common['q0'],np.float64)
        self.nxt=np.ascontiguousarray(np.maximum(self.graph['next'],0),dtype=np.int64);self.live=np.ascontiguousarray(~self.graph['terminal'],dtype=np.uint8)
        s,a=np.where(self.live);assert np.all(self.nxt[s,a]>s)
        self.lib=kernel();self.inputs=[self.wdir/p for p in ['COMMON_START.pt','graph.npz','schedule.npz','historical_rows.parquet','support_rows.parquet']]
        self.models=[self.actor,self.critic,self.target];self.optimizers=[self.ao,self.co]
        self.checkpoints=core.CHECKPOINTS;self.beta=core.AWAC_BETA;self.cap=core.WEIGHT_CLIP;self.clip=core.GRAD_CLIP

    def batch(self,u):
        if self.family=='KeyCorridor':return self.ref.batch_tensors(self.graph,self.h,self.qold,self.schedule['states'][u],self.schedule['times'][u],self.schedule['actions'][u])
        if self.family=='MiniGridLoCA':
            take=lambda k:torch.as_tensor(self.schedule[k][u],dtype=torch.long,device=self.device)
            return dict(cs=take('critic_s'),ca=take('critic_a'),s=take('actor_s'),a=take('actor_a'))
        i=self.offset+u
        return self.core.rows_to_batch(self.hist,self.support,self.schedule['row'][i],self.schedule['source'][i],self.graph)

    def critic_step(self,b):
        if self.family=='DeepSea':self.core.critic_step(self.actor,self.critic,self.target,self.co,b);return
        if self.family=='KeyCorridor' and self.cell['reference']=='exact':
            self.ref.critic_step(self.actor,self.critic,self.target,self.co,b);return
        if self.family=='KeyCorridor':
            with torch.no_grad():target=b['reward']+.97*(1-b['done'])*(self.actor(b['nx'],b['nmask']).exp()*self.target(b['nx'],b['ntf'])).sum(1)
            loss=F.mse_loss(self.critic(b['x'],b['tf']).gather(1,b['a'][:,None]).squeeze(1),target)
            self.co.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.critic.parameters(),self.clip);self.co.step();return
        s,a=b['cs'],b['ca'];ns=self.nxt_t[s,a]
        with torch.no_grad():
            lp=self.actor.log_policy(ns);p=lp.exp();v=[t(ns) for t in self.targets]
            q=torch.minimum(v[0],v[1])-self.alpha*lp if self.backbone=='SAC' else v[0]
            target=self.reward_t[s,a]+self.study.GAMMA*(1-self.done_t[s,a])*(p*q).sum(1)
        for c,opt in zip(self.critics,self.opts):
            loss=F.mse_loss(c(s).gather(1,a[:,None]).squeeze(1),target)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()

    @torch.no_grad()
    def values(self,b,native=False):
        if self.family=='KeyCorridor':
            p=self.ref.actor_policy(self.actor,self.features,self.mask)
            q=self.solver(p,b).clone();d=self.solver.output.clone()
            if native:q=self.ref.pdr_batch_q(self.actor,self.graph,self.h,self.qdevice,self.oldpi_t,b)[0];d=q-b['qold']
            if not native:self.revaluation_cache=(b['qold'],q,p)
            return b['qold'],d,q
        if self.family=='MiniGridLoCA':
            p=self.study.extract_policy(self.actor,self.states);g=self.g;gamma=self.study.GAMMA
            if native:d,_,_=self.core.generalized_pdr(g,self.oldpi,p,self.qold,gamma,self.alpha)
            else:
                aggregate=(p*(self.qold-self.alpha*np.log(np.clip(p,1e-15,1)))).sum(1)
                source=self.reward_delta.copy();source[self.rows,self.actions]+=gamma*(aggregate[self.succ]-self.oldaggregate[self.succ]);source[self.terminal]=0
                transition=sp.coo_matrix((p[self.rows,self.actions],(self.rows,self.succ)),shape=(g.n_states,g.n_states)).tocsc()
                v=spla.spsolve(self.identity-gamma*transition,(p*source).sum(1))
                d=source.copy();d[self.rows,self.actions]+=gamma*v[self.succ]
                propagated=np.zeros_like(d);propagated[self.rows,self.actions]=gamma*(p[self.succ]*d[self.succ]).sum(1)
                assert np.max(np.abs(d-source-propagated))<1e-10
            take=lambda x:torch.as_tensor(x,dtype=torch.float32,device=self.device)[b['s']]
            if not native:self.revaluation_cache=(self.qold,self.qold+d,p)
            return take(self.qold),take(d),take(self.qold+d)
        p=self.core.actor_policy(self.actor,self.graph)
        if native:q,_=self.core.pdr_q(p,self.oldpi,self.qold,self.graph)
        else:
            source=np.ascontiguousarray(((p-self.oldpi)*self.qold).sum(1),np.float64);pd=np.ascontiguousarray(p,np.float64);q=np.empty_like(self.qold)
            self.lib.deep_solve(len(p),ptr(self.nxt),ptr(self.live),ptr(pd),ptr(source),ptr(self.qold),ptr(q))
        take=lambda x:torch.as_tensor(x[b['s_id']],dtype=torch.float32,device=self.device)
        if not native:self.revaluation_cache=(self.qold,q,p)
        return take(self.qold),take(q-self.qold),take(q)

    @torch.no_grad()
    def td_values(self,b):
        if self.family=='KeyCorridor':return self.critic(b['x'],b['tf'])+(b['qold'] if self.cell['reference']=='exact' else 0.)
        if self.family=='DeepSea':return self.critic(b['x'])
        qs=[c(b['s']) for c in self.critics];return torch.minimum(qs[0],qs[1]) if self.backbone=='SAC' else qs[0]

    def logp(self,b):
        if self.family=='KeyCorridor':return self.actor(b['x'],b['mask'])
        if self.family=='MiniGridLoCA':return self.actor.log_policy(b['s'])
        return self.actor(b['x'])

    def objective(self,b,q):
        lp=self.logp(b);p=lp.exp();adv=q.detach()-(p.detach()*q.detach()).sum(1,keepdim=True)
        if self.backbone=='SAC':return (p*(self.alpha*lp-q.detach())).sum(1).mean(),adv,None
        if self.backbone=='EXPECTED_SARSA_QAC':return -(p.detach()*lp*adv).sum(1).mean(),adv,None
        # Detach the COMPLETE AWAC weight, including its policy baseline.
        w=torch.clamp(torch.exp(adv/self.beta),0.,self.cap).detach()
        return -(w.gather(1,b['a'][:,None])*lp.gather(1,b['a'][:,None])).mean(),adv,w

    def signal(self,b,arm):
        if arm=='TD':return self.td_values(b)
        _,d,q=self.values(b);return (d if arm.startswith('DELTA') else q)*(8. if arm.endswith('8') else 1.)

    def actor_step(self,b,q):
        loss,adv,w=self.objective(b,q);self.ao.zero_grad(set_to_none=True);loss.backward()
        if self.clip is not None:torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.clip)
        self.ao.step()
        if self.family=='MiniGridLoCA':
            for t,c in zip(self.targets,self.critics):self.study.polyak(t,c)
        elif self.family=='DeepSea':self.core.soft_update(self.target,self.critic)
        else:
            with torch.no_grad():
                for t,p in zip(self.target.parameters(),self.critic.parameters()):t.mul_(.995).add_(p,alpha=.005)
        return loss,adv,w

    def step(self,u,arm):
        b=self.batch(u);self.critic_step(b);q=self.signal(b,arm);return self.actor_step(b,q)

    def policy(self):
        if self.family=='KeyCorridor':return self.ref.actor_policy(self.actor,self.features,self.mask).cpu().numpy()
        if self.family=='MiniGridLoCA':return self.study.extract_policy(self.actor,self.states)
        return self.core.actor_policy(self.actor,self.graph)

    def evaluate(self):
        p=self.policy()
        if self.family=='KeyCorridor':return self.ref.exact_metrics(torch.as_tensor(p,device=self.device),self.graph,self.h)
        if self.family=='MiniGridLoCA':return self.core.evaluate_policy(self.g,p,self.study.GAMMA)
        return self.core.exact_metrics(p,self.graph)

    def state(self):
        return dict(models=[copy.deepcopy(m.state_dict()) for m in self.models],optimizers=[copy.deepcopy(o.state_dict()) for o in self.optimizers],
            rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate())
    def restore(self,z):
        for m,s in zip(self.models,z['models']):m.load_state_dict(s)
        # load_state_dict may reuse same-device Adam tensors. Training must never
        # mutate the immutable checkpoint used to initialize another branch.
        for o,s in zip(self.optimizers,z['optimizers']):o.load_state_dict(copy.deepcopy(s))
        torch.set_rng_state(z['rng_torch'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in z['rng_cuda']]);np.random.set_state(z['rng_numpy']);random.setstate(z['rng_python'])
