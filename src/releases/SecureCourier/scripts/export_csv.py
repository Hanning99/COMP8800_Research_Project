"""Export recorded scores, then calculate all figure statistics from the CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('STANDARD_TD', 'FRESH_PDR')
METRICS = ('p_shortcut', 'J_exact')
GRID = np.arange(0, 2049, 16)

def read_json(path):
    return json.loads(path.read_text())

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'data')
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    cohort=read_json(ROOT/'provenance/cohort.json')
    assert len(cohort)==50 and len({e['seed'] for e in cohort})==50
    rows=[]
    for event in cohort:
        for arm in ARMS:
            for cycle in GRID:
                source=Path('records/scored')/event['event']/arm/f'cycle_{cycle:06d}.json'
                raw=read_json(ROOT/source)
                assert raw['cycle']==int(cycle) and raw['arm']==arm
                instances=raw['instances']; assert len(instances)==16
                for metric in METRICS:
                    value=float(raw[metric])
                    np.testing.assert_allclose(value,np.mean([r[metric] for r in instances]),atol=1e-14,rtol=0)
                checkpoint=(Path('inputs/events')/event['event']/'capability_checkpoint' if cycle==0
                            else Path('checkpoints')/event['event']/arm/f'cycle_{cycle:06d}')
                rows.append(dict(event=event['event'],seed=event['seed'],arm=arm,cycle=int(cycle),
                    p_shortcut=raw['p_shortcut'],J_exact=raw['J_exact'],
                    full_start_shortcut=float(np.mean([r['full_start_shortcut_route_probability'] for r in instances])),
                    full_task_success=raw['capability']['full_task_success'],
                    gate_sequence_success=raw['capability']['gate_sequence_success'],
                    safe_route_success=raw['capability']['safe_route_success'],
                    source_score=str(source),checkpoint=str(checkpoint),checkpoint_sha256=raw['checkpoint_sha256']))
    raw_path=a.output/'per_seed.csv';write_csv(raw_path,rows)
    # Read the exported CSV back: downstream statistics have no JSON dependency.
    with raw_path.open(newline='') as f:
        records=list(csv.DictReader(f))
    keyed={(r['event'],r['arm'],int(r['cycle'])):r for r in records}
    assert len(keyed)==len(records)==50*2*129
    rng=np.random.default_rng(20260907)
    draws=rng.integers(0,50,(20000,50));counts=np.zeros((20000,50))
    np.add.at(counts,(np.repeat(np.arange(20000),50),draws.ravel()),1/50)
    def summary(x):
        means=counts@x
        return np.mean(x,axis=0),np.quantile(means,[.025,.975],axis=0)
    arrays={};curve_rows=[];seed_auc=[];stats_rows=[]
    thesis=read_json(ROOT/'provenance/thesis/statistics.json')
    historical_curves=read_json(ROOT/'provenance/thesis/plotted-summaries.json')['curves']
    for arm in ARMS:
        name='PDR' if arm=='FRESH_PDR' else arm
        for metric in METRICS:
            x=np.array([[float(keyed[(e['event'],arm,int(u))][metric]) for u in GRID] for e in cohort])
            assert x.shape==(50,129) and np.isfinite(x).all()
            arrays[arm,metric]=x
            mean,ci=summary(x)
            np.testing.assert_allclose(mean,historical_curves[name][metric]['mean'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(ci,historical_curves[name][metric]['ci95'],atol=1e-12,rtol=0)
            for i,u in enumerate(GRID):
                curve_rows.append(dict(arm=arm,cycle=int(u),metric=metric,mean=mean[i],ci95_low=ci[0,i],ci95_high=ci[1,i],n=50))
            auc=np.trapezoid(x[:,:65],GRID[:65],axis=1)/1024
            for e,v in zip(cohort,auc):
                seed_auc.append(dict(event=e['event'],seed=e['seed'],arm=arm,metric=metric,window_start=0,window_end=1024,normalized_auc=v))
            for stat,values in [('early_normalized_auc',auc),('cycle_1024',x[:,64]),('cycle_0',x[:,0])]:
                m,c=summary(values)
                stats_rows.append(dict(comparison=arm,metric=metric,statistic=stat,mean=m,ci95_low=c[0],ci95_high=c[1],n=50,positive_pairs=''))
    for metric in METRICS:
        np.testing.assert_array_equal(arrays[ARMS[0],metric][:,0],arrays[ARMS[1],metric][:,0])
        diff=arrays['FRESH_PDR',metric]-arrays['STANDARD_TD',metric]
        early=np.trapezoid(diff[:,:65],GRID[:65],axis=1)/1024
        for stat,values in [('early_normalized_auc',early),('cycle_1024',diff[:,64])]:
            m,c=summary(values)
            stats_rows.append(dict(comparison='PDR_minus_STANDARD_TD',metric=metric,statistic=stat,mean=m,ci95_low=c[0],ci95_high=c[1],n=50,positive_pairs=int((values>0).sum())))
            if stat=='early_normalized_auc':
                old=thesis['comparisons']['PDR_minus_STANDARD_TD'][metric]
                np.testing.assert_allclose([m,*c],[old['mean'],*old['ci95']],atol=1e-12,rtol=0)
        # Preserved disclosure: exclusion of event 000 is secondary sensitivity only.
        values=early[1:];rng49=np.random.default_rng(20260907)
        c=np.quantile(values[rng49.integers(0,49,(20000,49))].mean(1),[.025,.975])
        stats_rows.append(dict(comparison='PDR_minus_STANDARD_TD_excluding_event000',metric=metric,
            statistic='early_normalized_auc',mean=values.mean(),ci95_low=c[0],ci95_high=c[1],n=49,positive_pairs=int((values>0).sum())))
    locked=read_json(ROOT/'provenance/locked_results/SC_D_LOCKED_ANALYSIS.json')
    for metric in ['full_task_success','gate_sequence_success','safe_route_success']:
        values={arm:np.array([float(keyed[(e['event'],arm,2048)][metric]) for e in cohort]) for arm in ARMS}
        for name,x in [*values.items(),('PDR_minus_STANDARD_TD',values['FRESH_PDR']-values['STANDARD_TD'])]:
            m,c=summary(x)
            stats_rows.append(dict(comparison=name,metric=metric,statistic='cycle_2048',mean=m,
                ci95_low=c[0],ci95_high=c[1],n=50,positive_pairs=int((x>0).sum()) if name=='PDR_minus_STANDARD_TD' else ''))
            if name=='PDR_minus_STANDARD_TD':
                old=locked['metrics'][metric]
                np.testing.assert_allclose([m,*c],[old['mean'],*old['ci95']],atol=1e-12,rtol=0)
    write_csv(a.output/'curve_summary.csv',curve_rows)
    write_csv(a.output/'early_auc_per_seed.csv',seed_auc)
    write_csv(a.output/'reported_statistics.csv',stats_rows)
    metadata=dict(schema=1,primary_cohort=50,training_instances_per_seed=16,
        grid='0,16,...,2048',figure_window=[0,1024],pdr_active_cycles=[1,1024],
        discount=.97,horizon=40,bootstrap_seed=20260907,bootstrap_resamples=20000,
        intervals='Paired seed percentile bootstrap; curve intervals are pointwise, not simultaneous.',
        p_shortcut='Conditional policy probability of take_shortcut at junction, episode step 8; equally averaged over 16 training variants.',
        J_exact='Exact stochastic-policy expected discounted delivery return from depot, not greedy success rate.',
        early_auc='Trapezoidal integration over cycle 0..1024, divided by 1024; not endpoint return.',
        historical_agreement_atol=1e-12,all_historical_checks_passed=True,
        per_seed_csv_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Exported {len(records)} paired-cohort rows; all statistics and confidence intervals match the archived summaries.')

if __name__=='__main__':main()
