"""Compare a fresh continuation's model states and behaviour with its archive."""
import argparse
import json
from pathlib import Path
import bootstrap
from bootstrap import ROOT
import torch
from training_core import load_model_bundle
from pdr_awac_adapter import FactorizedPolicyEvaluator
from rescore import get_dag, fast_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    cfg=json.loads((a.run/'RUN_CONFIG.json').read_text())
    u=cfg['cycles'];e=cfg['event'];arm=cfg['arm']
    fresh=a.run/'checkpoints'/f'cycle_{u:06d}'
    archived=ROOT/'checkpoints'/e/arm/f'cycle_{u:06d}'
    x=torch.load(fresh/'state.pt',map_location='cpu',weights_only=False)
    y=torch.load(archived/'state.pt',map_location='cpu',weights_only=False)
    errors={key:max(float((x[key][k]-y[key][k]).abs().max()) for k in x[key]) for key in ['actor','critic','target']}
    actor,_,_,_,_,vocab,_,_=load_model_bundle(fresh,torch.device('cpu'))
    dag=get_dag();evaluator=FactorizedPolicyEvaluator(dag,vocab,torch.device('cpu'))
    actual=fast_metrics(actor,evaluator,dag)
    record=json.loads((ROOT/'records/scored'/e/arm/f'cycle_{u:06d}.json').read_text())
    metric_errors={k:abs(v-record[k]) for k,v in actual.items()}
    # Cross-device stochastic training need not be bitwise equal. These limits
    # apply only to the short engineering continuation, not a full-run promise.
    passed=max(errors.values())<2e-4 and max(metric_errors.values())<2e-5
    report=dict(status='PASS' if passed else 'NUMERICAL_MISMATCH',event=e,arm=arm,cycles=u,device=cfg['device'],
        max_abs_parameter_errors=errors,metric_errors=metric_errors,
        limitation='Short continuation validation; not a claim of bitwise full-cohort retraining on different hardware.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if not passed:raise SystemExit('Fresh continuation differs from the archived training trajectory; see the saved report.')

if __name__=='__main__':main()
