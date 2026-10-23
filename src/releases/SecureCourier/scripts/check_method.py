"""Verify the graph, learned reference arrays and differential solver locally."""
import argparse
import json
import runpy
from pathlib import Path
import bootstrap
from bootstrap import ROOT
import numpy as np
import torch
from scd_common import canonical_hash
from build_closed_dag import build_graph
from compile_pdr_reference import evaluate_actor_critic
from training_core import load_model_bundle
from reverse_pdr import ClosedDAG

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--event',default='confirmatory_000')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/method.json')
    a=p.parse_args();torch.set_num_threads(1)
    rebuilt=build_graph();saved=json.loads((ROOT/'inputs/support_graph.json').read_text())
    assert canonical_hash(rebuilt)==canonical_hash(saved)
    dag=ClosedDAG(ROOT/'inputs/support_graph.json')
    actor,critic,_,_,_,vocab,_,_=load_model_bundle(ROOT/'inputs/events'/a.event/'old/checkpoint',torch.device('cpu'))
    pi,q=evaluate_actor_critic(dag,actor,critic,vocab,torch.device('cpu'))
    r=ROOT/'inputs/references'/a.event
    errors=dict(pi_ref=float(np.max(np.abs(pi-np.load(r/'pi_ref.npy')))),
        qbar=float(np.max(np.abs(q-np.load(r/'qbar.npy')))))
    assert max(errors.values())<2e-6,errors
    tests=runpy.run_path(str(ROOT/'legacy/sc_d/tests/test_reverse_pdr.py'))
    for name,fn in tests.items():
        if name.startswith('test_'):fn()
    report=dict(status='PASS',rebuilt_graph_identical=True,reference_cpu_errors=errors,
        event=a.event,solver_tests='hand-computed differential, zero policy change, terminal boundary')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
