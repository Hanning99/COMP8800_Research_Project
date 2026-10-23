"""Portable continuation from the archived post-acquisition checkpoint.

Uses unchanged archived collection, replay, critic, PDR and actor functions.
Fresh outputs are separate from archived evidence. This is a reproduction run,
not a new confirmatory experiment or a re-run of capability acquisition.
"""
import argparse
import json
import os
from pathlib import Path
import bootstrap
from bootstrap import ROOT
import torch
from training_core import (ReplayBuffer, OFFLINE_BATCH, ONLINE_BATCH, ONLINE_CAPACITY,
    base_rows, clean_collate, collect_assisted_episode, critic_step, load_model_bundle,
    save_model_bundle, split_instances, update_target, append_jsonl)
from pdr_awac_adapter import PDRService, actor_weight_update
from scd_common import sha256, module_hash, tensor_tree_hash
from run_sc_d import pdr_intervention_active

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--event',required=True)
    p.add_argument('--arm',choices=['STANDARD_TD','FRESH_PDR'],required=True)
    p.add_argument('--cycles',type=int,default=2048)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert 1<=a.cycles<=2048
    allowed={e['event'] for e in json.loads((ROOT/'provenance/cohort.json').read_text())}
    assert a.event in allowed
    out=a.output.resolve()
    for protected in ['inputs','records','checkpoints','legacy','provenance','data','figures','scripts']:
        if out.is_relative_to(ROOT/protected):raise RuntimeError('Output overlaps archived evidence or release files')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output must be new or empty')
    # Verify input hashes against the release inventory before any training.
    sources=json.loads((ROOT/'provenance/source_files.json').read_text())
    prefixes=[f'inputs/events/{a.event}/',f'inputs/references/{a.event}/','inputs/support_graph.json','legacy/']
    for r in sources:
        if any(r['path'].startswith(prefix) for prefix in prefixes):
            assert sha256(ROOT/r['path'])==r['sha256'],r['path']
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device(a.device);source=ROOT/'inputs/events'/a.event
    actor,critic,target,ao,co,vocab,_,rng=load_model_bundle(source/'capability_checkpoint',device)
    initial={k:module_hash(m) for k,m in [('actor',actor),('critic',critic),('target',target)]}
    rows=base_rows(source);off=[r for r in rows if r['_scb_origin']=='offline'];on=[r for r in rows if r['_scb_origin']=='online']
    offline=ReplayBuffer(len(off)+1);offline.extend(off)
    online=ReplayBuffer(ONLINE_CAPACITY);online.extend(on)
    service=PDRService(ROOT/'inputs/support_graph.json',ROOT/'inputs/references'/a.event,vocab,device) if a.arm=='FRESH_PDR' else None
    config=dict(event=a.event,arm=a.arm,cycles=a.cycles,intervention_end=1024,
        device=str(device),initial_hashes=initial,initial_state_sha256=sha256(source/'capability_checkpoint/state.pt'),
        offline_online_batch=[OFFLINE_BATCH,ONLINE_BATCH],purpose='reproduction',scoring_during_training=False)
    (out/'RUN_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    total_rows=0
    for cycle in range(1,a.cycles+1):
        collected,episode=collect_assisted_episode(actor,vocab,split_instances()['train'],device,rng,cycle)
        for index,row in enumerate(collected):
            row.update(episode_id=f'continuation_c{cycle}',transition_id=f'continuation_c{cycle}_t{index}',
                _scb_uid=f'continuation:{cycle:06d}:{index:03d}',_scb_origin='continuation')
            online.add(row);append_jsonl(out/'continuation_stream.jsonl',row);total_rows+=1
        raw_batch=offline.sample(OFFLINE_BATCH,rng)+online.sample(ONLINE_BATCH,rng)
        batch=clean_collate(raw_batch,vocab,device)
        cm=critic_step(actor,critic,target,co,batch)
        active=pdr_intervention_active(a.arm,cycle,1024)
        refresh=service.refresh(actor) if active else None
        am=actor_weight_update(arm='FRESH_PDR' if active else 'STANDARD_TD',actor=actor,critic=critic,
            optimizer=ao,batch=batch,raw_rows=raw_batch,pdr_service=service,refresh=refresh)
        update_target(target,critic)
        append_jsonl(out/'training_log.jsonl',dict(cycle=cycle,episode=episode,**cm,**am))
        if cycle%16==0 or cycle in [1,1025,a.cycles]:
            save_model_bundle(out/'checkpoints'/f'cycle_{cycle:06d}',actor,critic,target,ao,co,vocab,
                dict(cycle=cycle,arm=a.arm,matching_actor_critic=True,continuation_rows=total_rows),rng,device)
            print(f'{a.event} {a.arm} {cycle}/{a.cycles}',flush=True)
    final=dict(cycle=a.cycles,actor_hash=module_hash(actor),critic_hash=module_hash(critic),
        target_hash=module_hash(target),actor_optimizer_hash=tensor_tree_hash(ao.state_dict()),
        critic_optimizer_hash=tensor_tree_hash(co.state_dict()))
    (out/'RUN_MANIFEST.json').write_text(json.dumps(dict(status='REPRODUCTION_COMPLETE',config=config,final=final),indent=2)+'\n')

if __name__=='__main__':main()
