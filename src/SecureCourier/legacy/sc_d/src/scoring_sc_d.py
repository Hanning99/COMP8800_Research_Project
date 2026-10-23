"""Post-training scoring of immutable checkpoints; no optimizer updates."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from scd_common import PARENT, TEMPERATURE, WEIGHT_CLIP, sha256, clean
from training_core import base_rows, clean_collate, load_model_bundle, split_instances
from pdr_awac_adapter import PDRService, masked_log_softmax, center_under_policy
from scabc_common import exact_instance_metrics, capability_macro
from oracle_scoring import exact_rows


def signal(actor, critic, vocab, device, raw_rows, service, refresh):
    batch = clean_collate(raw_rows, vocab, device)
    logp = masked_log_softmax(actor(batch['obs_tokens'], batch['act_tokens']), batch['action_mask'])
    if refresh is None:
        with torch.no_grad():
            q = critic(batch['obs_tokens'], batch['act_tokens'])
    else:
        q, mask, _ = service.lookup(raw_rows, refresh.result.q_pdr_by_pair, device, logp.dtype)
        assert torch.equal(mask, batch['action_mask'])
    q = torch.where(batch['action_mask'], q, torch.zeros_like(q)).detach()
    adv = center_under_policy(q, logp.exp(), batch['action_mask']).detach()
    selected = adv.gather(1, batch['action_idx'][:, None]).squeeze(1)
    w = torch.exp(selected / TEMPERATURE).clamp(max=WEIGHT_CLIP).detach()
    scored = exact_rows(actor, vocab, device, raw_rows)
    by_uid = {x['uid']: x for x in scored}
    exact = torch.tensor([by_uid[r['_scb_uid']]['weight_exact'] for r in raw_rows], device=device, dtype=w.dtype)
    chosen = logp.gather(1, batch['action_idx'][:, None]).squeeze(1)
    params = tuple(actor.parameters())
    g = torch.cat([v.reshape(-1) for v in torch.autograd.grad(-(w * chosen).mean(), params, retain_graph=True)])
    ge = torch.cat([v.reshape(-1) for v in torch.autograd.grad(-(exact * chosen).mean(), params)])
    gn, en = float(g.norm()), float(ge.norm())
    return {
        'weight_error': float((w-exact).abs().mean()),
        'log_weight_error': float((w.clamp_min(1e-12).log()-exact.clamp_min(1e-12).log()).abs().mean()),
        'gradient_cosine': float(torch.dot(g,ge)/(g.norm()*ge.norm())) if min(gn,en)>=1e-12 else None,
        'gradient_norm': gn, 'exact_gradient_norm': en,
        'mean_advantage': float(selected.mean()),
    }


def score_checkpoint(checkpoint, source, panels, reference, graph, arm, cycle, device):
    actor, critic, _, _, _, vocab, _, _ = load_model_bundle(checkpoint, device)
    actor.eval(); critic.eval()
    all_rows = {r['_scb_uid']: r for r in base_rows(source)}
    h = json.loads((panels/'H_PANEL.json').read_text())['items']
    h_rows = [all_rows[x['uid']] for x in h]
    diag = [all_rows[x] for x in json.loads((panels/'DIAGNOSTIC_BATCH.json').read_text())]
    service = PDRService(graph, reference, vocab, device) if arm=='FRESH_PDR' else None
    refresh = service.refresh(actor) if service and cycle<=1024 else None
    gradient = signal(actor,critic,vocab,device,diag,service,refresh)
    h_signal = signal(actor,critic,vocab,device,h_rows,service,refresh) if h_rows else None
    instances = [exact_instance_metrics(actor,vocab,i,device) for i in split_instances()['train']]
    capability = capability_macro(actor,vocab,split_instances()['train'],device)['macro']
    return clean({
        'cycle': cycle, 'arm': arm, 'scoring_only': True,
        'checkpoint_sha256': sha256(checkpoint/'state.pt'),
        'J_exact': np.mean([r['J_exact'] for r in instances]),
        'log_odds': np.mean([math.log(max(r['p_shortcut'],1e-12)/max(r['p_safe'],1e-12)) for r in instances]),
        'p_shortcut': np.mean([r['p_shortcut'] for r in instances]),
        'weight_error': h_signal['weight_error'] if h_signal else None,
        'log_weight_error': h_signal['log_weight_error'] if h_signal else None,
        'gradient_cosine': gradient['gradient_cosine'],
        'gradient_norm': gradient['gradient_norm'], 'exact_gradient_norm': gradient['exact_gradient_norm'],
        'capability': capability, 'instances': instances,
        'correction_linf': float(np.abs(refresh.result.correction_by_pair).max()) if refresh else 0.,
        'h_mean_advantage': h_signal['mean_advantage'] if h_signal else None,
    })


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--event',required=True); p.add_argument('--arm',required=True,choices=['FRESH_PDR','STANDARD_TD'])
    p.add_argument('--device',default='cpu'); args=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    from lock_sc_d import verify_lock
    verify_lock(root)
    source=PARENT/'source_events'/args.event
    panels=PARENT/'manifests/sc_c_panels'/args.event
    run=root/'runs'/args.event/args.arm
    if args.arm=='FRESH_PDR':
        assert json.loads((run/'RUN_MANIFEST.json').read_text())['status']=='SEALED_COMPLETE'
    else:
        run=PARENT/'sc_c_oracle'/args.event/'STALE'
    output=root/'scored'/args.event/args.arm
    output.mkdir(parents=True,exist_ok=True)
    cycles=sorted(set(range(0,2049,16))|({1,1025} if args.arm=='FRESH_PDR' else set()))
    for cycle in cycles:
        path=output/f'cycle_{cycle:06d}.json'
        if path.exists(): continue
        checkpoint=source/'capability_checkpoint' if cycle==0 else run/'checkpoints'/f'cycle_{cycle:06d}'
        result=score_checkpoint(checkpoint,source,panels,root/'references'/args.event,root/'manifests/support_graph.json',args.arm,cycle,torch.device(args.device))
        path.write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps({'cycles':cycles,'scoring_only':True,'files':{f.name:sha256(f) for f in output.glob('cycle_*.json')}})+'\n')

if __name__=='__main__': main()
