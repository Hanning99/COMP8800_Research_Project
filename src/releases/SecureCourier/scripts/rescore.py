"""Recompute the two paper metrics from full saved checkpoints (CPU supported)."""
import argparse
import csv
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import bootstrap
from bootstrap import ROOT
import numpy as np
import torch
from training_core import Actor, split_instances, EMBED_DIM, HIDDEN_DIM
from pdr_awac_adapter import FactorizedPolicyEvaluator
from reverse_pdr import ClosedDAG

DAG=None
def get_dag():
    global DAG
    if DAG is None:DAG=ClosedDAG(ROOT/'inputs/support_graph.json')
    return DAG

def fast_metrics(actor,evaluator,dag):
    """Exact finite-horizon DP using the archived factorized policy evaluator.

    The observation score is common to all actions and cancels in softmax.
    This is algebraically the native policy, up to float32 roundoff; --native
    independently invokes the original unfactorized evaluator.
    """
    pi=evaluator.evaluate(actor)
    rewards=np.array([r['reward'] for r in dag.pairs])
    values=np.zeros(len(dag.states))
    for step in range(39,-1,-1):
        ids=dag.step_pairs[step];q=rewards[ids].copy();live=dag.continuation[ids]
        q[live]+=.97*values[dag.successor[ids[live]]]
        ids_s=dag.step_states[step]
        values[ids_s]=np.bincount(dag.pair_state[ids],weights=pi[ids]*q,minlength=len(values))[ids_s]
    names=sorted({s['instance_id'] for s in dag.states})
    starts=[dag.state_index[f'{s}\x1fdepot\x1f00'] for s in names]
    choices=[dag.pair_index[f'{s}\x1fjunction\x1f08\x1ftake_shortcut'] for s in names]
    return dict(J_exact=float(values[starts].mean()),p_shortcut=float(pi[choices].mean()))

def run_task(task):
    torch.set_num_threads(1)
    event,arm,cycles,native,atol,run=task
    initial=ROOT/'inputs/events'/event/'capability_checkpoint'
    with (initial/'vocab.pkl').open('rb') as f:vocab=pickle.load(f)
    actor=Actor(vocab.size,EMBED_DIM,HIDDEN_DIM).eval()
    dag=get_dag();evaluator=FactorizedPolicyEvaluator(dag,vocab,torch.device('cpu'))
    result=[]
    for cycle in cycles:
        path=initial if cycle==0 else ((Path(run)/'checkpoints'/f'cycle_{cycle:06d}') if run else ROOT/'checkpoints'/event/arm/f'cycle_{cycle:06d}')
        recorded=json.loads((ROOT/'records/scored'/event/arm/f'cycle_{cycle:06d}.json').read_text())
        state=path/'state.pt';h=hashlib.sha256(state.read_bytes()).hexdigest()
        if run is None:assert h==recorded['checkpoint_sha256'],str(state)
        else:
            manifest=json.loads((path/'manifest.json').read_text())
            assert h==manifest['state_sha256'],str(state)
        payload=torch.load(state,map_location='cpu',weights_only=False)
        actor.load_state_dict(payload['actor'])
        if native:
            from scabc_common import exact_instance_metrics
            with torch.inference_mode():
                instances=[exact_instance_metrics(actor,vocab,i,torch.device('cpu')) for i in split_instances()['train']]
            values={k:float(np.mean([r[k] for r in instances])) for k in ['J_exact','p_shortcut']}
        else:values=fast_metrics(actor,evaluator,dag)
        row=dict(event=event,arm=arm,cycle=cycle,**values,checkpoint_sha256=h,evaluator='native' if native else 'factorized')
        for k,v in values.items():
            error=abs(v-recorded[k]);row[k+'_abs_error']=error
            if run is None and error>atol:raise AssertionError((event,arm,cycle,k,v,recorded[k],error,atol))
        result.append(row)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events',nargs='+',default=['all'])
    p.add_argument('--arms',nargs='+',choices=['STANDARD_TD','FRESH_PDR'],default=['STANDARD_TD','FRESH_PDR'])
    p.add_argument('--cycles',nargs='+',type=int)
    p.add_argument('--end',type=int,choices=[1024,2048],default=1024)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--native',action='store_true',help='Original unfactorized policy evaluator (slower).')
    p.add_argument('--run',type=Path,help='Evaluate a new train.py run; archive differences are reported, not required to vanish.')
    p.add_argument('--atol',type=float,default=2e-6)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/rescored.csv')
    a=p.parse_args()
    events=[e['event'] for e in json.loads((ROOT/'provenance/cohort.json').read_text())] if a.events==['all'] else a.events
    cycles=a.cycles if a.cycles is not None else list(range(0,a.end+1,16))
    assert all(0<=u<=2048 for u in cycles)
    if a.run:
        cfg=json.loads((a.run/'RUN_CONFIG.json').read_text());events=[cfg['event']];a.arms=[cfg['arm']]
        if a.cycles is None:
            cycles=[0]+sorted(int(p.name.split('_')[-1]) for p in (a.run/'checkpoints').glob('cycle_*') if int(p.name.split('_')[-1])%16==0)
    tasks=[(e,arm,cycles,a.native,a.atol,str(a.run.resolve()) if a.run else None) for e in events for arm in a.arms]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_task,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            rows.extend(f.result())
            print(f'Scored {i}/{len(tasks)} event/arm groups',flush=True)
    rows.sort(key=lambda r:(r['event'],r['arm'],r['cycle']))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(checkpoints=len(rows),events=len(events),cycles=cycles,atol=a.atol,
        evaluator='native' if a.native else 'factorized',all_checkpoint_hashes_match=True,
        checkpoint_hash_source='run manifests' if a.run else 'original score records',
        archive_metric_agreement_required=a.run is None,
        max_abs_errors={k:max(r[k+'_abs_error'] for r in rows) for k in ['p_shortcut','J_exact']})
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
