"""Actor interventions; reference solves and critic updates are unchanged."""
from learner import *

CONTROLS = ['FULL1', 'FULL8', 'DELTA8']
H1 = ['REF1_DELTA8', 'REF2_DELTA8', 'REF4_DELTA8']
H2 = ['MIX_RAW50', 'MIX_NORM50']
H2_CONTROLS = ['FULL8_NORM', 'DELTA8_NORM', 'ZERO']

def arms_for(cell):
    return CONTROLS + H1 + (H2 + H2_CONTROLS if cell['backbone'] == 'AWAC' else [])

def actor_objective(lp, actions, old, delta, corrected, arm, backbone, beta, cap, entropy):
    p = lp.exp()
    old, delta, corrected = old.detach(), delta.detach(), corrected.detach()
    center = lambda v: v - (p.detach() * v).sum(1, keepdim=True)
    ref_adv, delta_adv = center(old), center(delta)
    full8, delta8 = center(8 * corrected), center(8 * delta)
    info = dict(reference_advantage=ref_adv, delta_advantage=delta_adv,
                full8_advantage=full8, delta8_advantage=delta8)
    if arm in H2 + ['FULL8_NORM', 'DELTA8_NORM']:
        assert backbone == 'AWAC'
        wf = torch.exp(full8 / beta).clamp(0., cap).detach()
        wd = torch.exp(delta8 / beta).clamp(0., cap).detach()
        mf = wf.gather(1, actions[:, None]).mean().detach()
        md = wd.gather(1, actions[:, None]).mean().detach()
        assert mf > 0 and md > 0, 'Cannot normalize an all-zero sampled weight vector'
        info.update(full_raw_weight=wf, delta_raw_weight=wd,
                    full_normalizer=mf, delta_normalizer=md)
        # Normalize AFTER the native cap, over sampled records. No second cap.
        if arm == 'MIX_RAW50': w = .5 * wf + .5 * wd
        elif arm == 'MIX_NORM50': w = .5 * (wf / mf) + .5 * (wd / md)
        elif arm == 'FULL8_NORM': w = wf / mf
        else: w = wd / md
        w = w.detach()
        loss = -(w.gather(1, actions[:, None]) * lp.gather(1, actions[:, None])).mean()
        # This is only a log-weight equivalent, not a centered Q advantage.
        signal = beta * w.clamp_min(torch.finfo(w.dtype).tiny).log()
        info['signal_kind'] = 'log_weight_equivalent'
        return loss, signal, w, info
    if arm == 'FULL1': value = corrected
    elif arm == 'FULL8': value = 8 * corrected
    elif arm == 'DELTA8': value = 8 * delta
    elif arm == 'ZERO': value = torch.zeros_like(corrected)
    elif arm in H1: value = float(arm[3]) * old + 8 * delta
    else: raise ValueError(arm)
    adv = center(value).detach()
    info['signal_kind'] = 'centered_advantage'
    if backbone == 'SAC':
        return (p * (entropy * lp - value)).sum(1).mean(), adv, None, info
    if backbone == 'EXPECTED_SARSA_QAC':
        return -(p.detach() * lp * adv).sum(1).mean(), adv, None, info
    w = torch.exp(adv / beta).clamp(0., cap).detach()
    return -(w.gather(1, actions[:, None]) * lp.gather(1, actions[:, None])).mean(), adv, w, info

class HypothesisLearner(Learner):
    def select(self, b, arm, old, delta, corrected):
        return dict(arm=arm, old=old, delta=delta, corrected=corrected)

    def signal(self, b, arm):
        old, delta, corrected = self.values(b)
        return self.select(b, arm, old, delta, corrected)

    def objective(self, b, signal):
        loss, adv, weights, info = actor_objective(
            self.logp(b), b['a'], signal['old'], signal['delta'], signal['corrected'],
            signal['arm'], self.backbone, self.beta, self.cap, getattr(self, 'alpha', 0.))
        self.signal_info = info
        return loss, adv, weights

def cosine(a, b):
    den = float(a.double().norm() * b.double().norm())
    return float(torch.dot(a.double(), b.double()) / den) if den > 1e-30 else None

def diagnostic(l, u, active_arm):
    b = l.batch(min(u, l.updates - 1))
    old, delta, corrected = l.values(b)
    lp = l.logp(b).detach()
    mask = b.get('mask', torch.ones_like(lp, dtype=torch.bool))
    chosen = lambda v: v.gather(1, b['a'][:, None]).squeeze(1)
    arrays = dict(old=old.cpu().numpy(), delta=delta.cpu().numpy(),
                  corrected=corrected.cpu().numpy(), policy=lp.exp().cpu().numpy(),
                  actions=b['a'].cpu().numpy(), valid=mask.cpu().numpy())
    if l.family == 'KeyCorridor':
        arrays.update(states=b['s'].cpu().numpy(), times=b['t'].cpu().numpy())
    elif l.family == 'MiniGridLoCA': arrays['states'] = b['s'].cpu().numpy()
    else: arrays['states'] = np.asarray(b['s_id'])
    rows, grads = [], {}
    for arm in arms_for(l.cell):
        loss, sig, w = l.objective(b, l.select(b, arm, old, delta, corrected))
        grad = flat_gradient(loss, l.actor)
        assert torch.isfinite(grad).all() and torch.isfinite(sig).all()
        grads[arm] = grad
        info = l.signal_info
        ref, da = info['reference_advantage'], info['delta_advantage']
        row = dict(update=u, input_arm=arm, active_arm=active_arm,
                   signal_kind=info['signal_kind'], signal_rms=float(chosen(sig).square().mean().sqrt()),
                   gradient_norm=float(grad.double().norm()),
                   reference_advantage_rms=float(chosen(ref).square().mean().sqrt()),
                   delta_advantage_rms=float(chosen(da).square().mean().sqrt()),
                   ref_delta_cosine=cosine(chosen(ref), chosen(da)))
        if w is not None:
            sw = chosen(w)
            row.update(weight_mean=float(sw.mean()), weight_max=float(sw.max()),
                       weight_ess_fraction=float(sw.sum().square() / (sw.square().sum() * len(sw))),
                       weight_at_or_above_native_cap_fraction=float((sw >= l.cap).float().mean()))
            arrays[f'{arm}_weights'] = w.cpu().numpy()
        for name in ['full_normalizer', 'delta_normalizer']:
            if name in info: row[name] = float(info[name])
        arrays[f'{arm}_signal'] = sig.cpu().numpy()
        rows.append(row)
    for row in rows:
        g = grads[row['input_arm']]
        row.update(gradient_cosine_full8=cosine(g, grads['FULL8']),
                   gradient_cosine_delta8=cosine(g, grads['DELTA8']))
    return rows, arrays
