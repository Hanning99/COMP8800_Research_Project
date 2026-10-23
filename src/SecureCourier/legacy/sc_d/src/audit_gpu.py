"""Disjoint CUDA preflight for the exact batch policy path and closed loop."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from scd_common import PARENT,sha256,write_pretty_json
from training_core import load_model_bundle
from reverse_pdr import ClosedDAG
from compile_pdr_reference import evaluate_actor_critic
from pdr_awac_adapter import FactorizedPolicyEvaluator

root=Path(__file__).resolve().parent.parent
source=PARENT/'pilots/m3_resume_a'
device=torch.device('cuda:0')
actor,critic,_,_,_,vocab,_,_=load_model_bundle(source/'capability_checkpoint',device)
dag=ClosedDAG(root/'manifests/support_graph.json')
pi,_=evaluate_actor_critic(dag,actor,critic,vocab,device)
fast=FactorizedPolicyEvaluator(dag,vocab,device).evaluate(actor)
err=float(np.max(np.abs(pi-fast)))
assert err<=5e-7,err
out=root/'pilots/amended_cuda_final'
subprocess.run([sys.executable,str(root/'src/run_sc_d.py'),'--source-event',str(source),'--reference',str(root/'pilots/d2_reference'),'--graph',str(root/'manifests/support_graph.json'),'--arm','FRESH_PDR','--output',str(out),'--device','cuda:0','--mode','pilot','--max-cycles','16','--disable-scoring'],check=True)
checks=[json.loads(s) for s in (out/'training_log.jsonl').read_text().splitlines()]
assert len(checks)==16 and all(r['weights_detached'] and r['padded_action_mass']==0 and r['lookup_coverage']==r['real_action_slots'] for r in checks)
write_pretty_json(root/'audits/GPU_PREFLIGHT.json',{'pass':True,'max_abs_policy_error':err,'gpu':torch.cuda.get_device_name(0),'code_hashes':{p.name:sha256(p) for p in (root/'src').glob('*.py')}})
print('CUDA preflight PASS',flush=True)
