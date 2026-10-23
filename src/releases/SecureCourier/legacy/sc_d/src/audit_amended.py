"""Validate amended source guards, current runner, solver and separate scorer."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from scd_common import PARENT,sha256,write_pretty_json
from training_core import load_model_bundle
from pdr_awac_adapter import FactorizedPolicyEvaluator
from compile_pdr_reference import evaluate_actor_critic
from reverse_pdr import ClosedDAG

ROOT=Path(__file__).resolve().parent.parent
SOURCE=PARENT/'pilots/m3_resume_a'
def run(name,arm='FRESH_PDR',cycles=4,extra=(),source=SOURCE,expect=None):
    out=ROOT/'pilots'/('amended_'+os.environ.get('SC_D_AUDIT_ATTEMPT','initial')+'_'+name)
    cmd=[sys.executable,str(ROOT/'src/run_sc_d.py'),'--source-event',str(source),'--reference',str(ROOT/'pilots/d2_reference'),'--graph',str(ROOT/'manifests/support_graph.json'),'--arm',arm,'--output',str(out),'--device','cpu','--mode','pilot','--max-cycles',str(cycles),'--checkpoint-interval','2','--evaluation-interval','2','--disable-scoring',*extra]
    result=subprocess.run(cmd,capture_output=True,text=True)
    if expect:
        assert result.returncode!=0 and expect in result.stderr, result.stderr
        assert not out.exists()
    else: assert result.returncode==0,result.stderr
    return out

def main():
    checks={}
    run('forbidden_source',source=PARENT/'source_events/confirmatory_000',expect='PILOT_SOURCE_MUST_BE_DISJOINT_PARENT_PILOT')
    checks['official_source_rejected_in_pilot']=True
    run('mismatched_reference',source=PARENT/'pilots/m3_resume_b',expect='PDR_REFERENCE_SOURCE_EVENT_MISMATCH')
    checks['reference_cross_event_rejected']=True
    full=run('resume_full')
    split=run('resume_split',extra=['--pause-after-cycle','2'])
    run('resume_split',extra=['--resume'])
    for name in ['training_log.jsonl','continuation_stream.jsonl','checkpoints/cycle_000004/state.pt']:
        assert sha256(full/name)==sha256(split/name)
    checks['resume_byte_identity']=True
    standard=run('standard_16',arm='STANDARD_TD',cycles=16)
    parent=ROOT/'pilots/d3_parent_standard_16/STALE'
    ours=json.loads((standard/'RUN_MANIFEST.json').read_text())['final']
    theirs=json.loads((parent/'checkpoints/cycle_000016/manifest.json').read_text())
    assert all(ours[k]==theirs[k] for k in ['actor_hash','critic_hash','target_hash','actor_optimizer_hash','critic_optimizer_hash'])
    assert sha256(standard/'continuation_stream.jsonl')==sha256(parent/'continuation_stream.jsonl')
    checks['standard_parent_parity']=True
    pdr=run('pdr_16',cycles=16)
    inactive=run('inactive',extra=['--intervention-end','0'])
    std4=run('standard4',arm='STANDARD_TD')
    assert sha256(inactive/'training_log.jsonl')==sha256(std4/'training_log.jsonl')
    checks['restoration_path']=True
    from run_sc_d import pdr_intervention_active
    assert pdr_intervention_active('FRESH_PDR',1024,1024) and not pdr_intervention_active('FRESH_PDR',1025,1024)
    actor,critic,_,_,_,vocab,_,_=load_model_bundle(SOURCE/'capability_checkpoint',torch.device('cpu'))
    dag=ClosedDAG(ROOT/'manifests/support_graph.json')
    native,_=evaluate_actor_critic(dag,actor,critic,vocab,torch.device('cpu'))
    fast=FactorizedPolicyEvaluator(dag,vocab,torch.device('cpu')).evaluate(actor)
    error=float(np.max(np.abs(native-fast)))
    assert error<=5e-7
    checks['batched_policy_matches_native']=True
    from scoring_sc_d import score_checkpoint
    before=sha256(SOURCE/'capability_checkpoint/state.pt')
    score=score_checkpoint(SOURCE/'capability_checkpoint',SOURCE,SOURCE/'sc_c_panels',ROOT/'pilots/d2_reference',ROOT/'manifests/support_graph.json','FRESH_PDR',0,torch.device('cpu'))
    assert before==sha256(SOURCE/'capability_checkpoint/state.pt')
    assert score['scoring_only'] and score['weight_error'] is not None and score['gradient_cosine'] is not None
    # No pilot score values are output, selected, or used to change thresholds.
    checks['separate_scorer_complete_and_checkpoint_immutable']=True
    sys.path.insert(0,str(ROOT/'tests'))
    import test_reverse_pdr
    for name in dir(test_reverse_pdr):
        if name.startswith('test_'): getattr(test_reverse_pdr,name)()
    checks['solver_micro_tests']=True
    from analyze_sc_d import auc,summary
    assert abs(auc({i:{'J_exact':i/1024} for i in range(0,1025,16)},'J_exact')-.5)<1e-12
    assert summary([1.]*50)['ci95']==[1.,1.]
    checks['statistics_synthetic_tests']=True
    out={'pass':all(checks.values()),'checks':checks,'prelock_deviation_disclosed':True,'batched_policy_max_abs_error':error,'code_hashes':{p.name:sha256(p) for p in (ROOT/'src').glob('*.py')}}
    write_pretty_json(ROOT/'audits/AMENDED_PREEXECUTION_AUDIT.json',out)
    print('Amended audit PASS',flush=True)

if __name__=='__main__':main()
