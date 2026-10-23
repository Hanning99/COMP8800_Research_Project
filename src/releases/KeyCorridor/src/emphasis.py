"""Independent reference/change coefficients; fixed AWAC temperature."""
from learner import *

class EmphasisLearner(Learner):
    def signal(self,b,config):
        old,d,q=self.values(b)
        a,beta=config['alpha'],config['beta']
        # Retain archived arithmetic for the common equal-coefficient controls.
        value=a*q if a==beta else a*old+beta*d
        return dict(value=value,old=old,delta=d,corrected=q,config=config)

    def objective(self,b,signal):
        assert self.family=='KeyCorridor' and self.backbone=='AWAC'
        lp=self.logp(b);p=lp.detach().exp();c=signal['config']
        value=signal['value'].detach()
        adv=(value-(p*value).sum(1,keepdim=True)).detach()
        # Keep the archived exp-then-cap convention; even saturated weights are detached.
        w=torch.exp(adv/c['temperature']).clamp(0.,self.cap).detach()
        loss=-(w.gather(1,b['a'][:,None])*lp.gather(1,b['a'][:,None])).mean()
        self.last_signal=signal;self.last_advantage=adv;self.last_weights=w
        return loss,adv,w
