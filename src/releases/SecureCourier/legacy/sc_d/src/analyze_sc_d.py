"""Locked paired-event inference and explicit prelock-exposure sensitivity."""
import json
from pathlib import Path
import numpy as np
from scd_common import clean, PARENT
from verdict_sc_d import verdict

SEED=20260907
RESAMPLES=20000
KEYS=('J_exact','log_odds','weight_error','gradient_cosine')
REL=('gate_sequence_success','safe_route_success','full_task_success')

def auc(rows, key, end=1024):
    grid=list(range(0,end+1,16))
    values=[rows.get(u,{}).get(key) for u in grid]
    good=np.asarray([v is not None and np.isfinite(v) for v in values])
    if key=='gradient_cosine':
        if good.mean()<.9 or not good[0] or not good[-1]: return None
        # Prelocked interpolation only across interior undefined cosine points.
        values=np.interp(grid,np.asarray(grid)[good],np.asarray(values,dtype=float)[good])
    elif not good.all(): return None
    return float(np.trapz(values,grid)/end)

def summary(values, seed=SEED):
    x=np.asarray([v for v in values if v is not None],dtype=float)
    if not len(x): return {'n':0,'mean':None,'median':None,'positive':0,'ci95':[None,None],'lower_one_sided_95':None}
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(RESAMPLES,len(x)))].mean(1)
    signs=rng.choice([-1,1],(RESAMPLES,len(x)))
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'positive':int((x>0).sum()),'ci95':np.quantile(means,[.025,.975]).tolist(),'lower_one_sided_95':float(np.quantile(means,.05)), 'paired_randomization_p_two_sided':float((1+(np.abs((signs*x).mean(1))>=abs(x.mean())).sum())/(RESAMPLES+1))}

def reduce_arm(root,event,arm):
    path=root/'scored'/event/arm
    if not (path/'COMPLETE.json').exists(): return None
    rows={int(x['cycle']):x for p in path.glob('cycle_*.json') for x in [json.loads(p.read_text())]}
    result={key:auc(rows,key) for key in KEYS}
    result.update({key:rows.get(2048,{}).get('capability',{}).get(key) for key in REL})
    result['full_return_auc']=auc(rows,'J_exact',2048)
    result['endpoint_return']=rows.get(2048,{}).get('J_exact')
    grid=list(range(0,2049,16))
    crossing=next((u for u,v in zip(grid,grid[1:]) if rows.get(u,{}).get('p_shortcut',-1)>=.5 and rows.get(v,{}).get('p_shortcut',-1)>=.5),None)
    result['adoption_time']=crossing
    result['adoption_censored']=crossing is None
    result['crossed_by_1024']=crossing is not None and crossing+16<=1024
    result['crossed_by_2048']=crossing is not None
    return result

def main():
    root=Path(__file__).resolve().parent.parent
    from lock_sc_d import verify_lock
    verify_lock(root)
    # A worker terminal ledger must account for every event before analysis.
    terminal=json.loads((root/'logs/COHORT_TERMINAL.json').read_text())
    assert len(terminal['events'])==50
    events=[]
    for i in range(50):
        event=f'confirmatory_{i:03d}'
        arms={a:reduce_arm(root,event,a) for a in ('FRESH_PDR','STANDARD_TD')}
        if all(arms.values()):
            scored={a:{int(x['cycle']):x for f in (root/'scored'/event/a).glob('cycle_*.json') for x in [json.loads(f.read_text())]} for a in arms}
            joint=[u for u in range(0,1025,16) if all(scored[a].get(u,{}).get('gradient_cosine') is not None for a in arms)]
            for a in arms:
                if len(joint)<.9*65 or 0 not in joint or 1024 not in joint:
                    arms[a]['gradient_cosine']=None
                else:
                    common={u:dict(v,gradient_cosine=v.get('gradient_cosine') if u in joint else None) for u,v in scored[a].items()}
                    arms[a]['gradient_cosine']=auc(common,'gradient_cosine')
        diffs={k:arms['FRESH_PDR'][k]-arms['STANDARD_TD'][k] if all(arms[a] is not None and arms[a].get(k) is not None for a in arms) else None for k in KEYS+REL}
        events.append({'event_id':event,'arms':arms,'differences':diffs})
    metrics={k:summary([e['differences'][k] for e in events]) for k in KEYS+REL}
    completeness={k:metrics[k]['n'] for k in KEYS+REL}
    completeness['joint_primary']=sum(all(e['differences'][k] is not None for k in KEYS+REL) for e in events)
    bounds={'best':[],'worst':[]}
    for event in events:
        p=(event['arms']['FRESH_PDR'] or {}).get('J_exact')
        s=(event['arms']['STANDARD_TD'] or {}).get('J_exact')
        bounds['best'].append((1 if p is None else p)-(0 if s is None else s))
        bounds['worst'].append((0 if p is None else p)-(1 if s is None else s))
    payload={'schema':'SC-D amended locked analysis','verdict':verdict(metrics,completeness), 'parent_verdict_unchanged':'SECURECOURIER_MECHANISM_CHAIN_CONFIRMED','events':events,'metrics':metrics,'completeness':completeness,'missingness_bounds':{k:{'mean':np.mean(v),'positive':sum(x>0 for x in v)} for k,v in bounds.items()},'exclude_preexposed_event000_sensitivity':{k:summary([e['differences'][k] for e in events[1:]]) for k in KEYS+REL},'prelock_exposure_disclosed':True,'bootstrap_seed':SEED,'resamples':RESAMPLES}
    out=root/'results/SC_D_LOCKED_ANALYSIS.json'
    calibration=[]
    for event in events:
        e=event['event_id'];p=event['arms']['FRESH_PDR'];s=event['arms']['STANDARD_TD']
        trace=json.loads((PARENT/'sc_c_oracle'/e/'CORRECT/evaluation_trace.json').read_text())
        cr={int(r['cycle']):{'J_exact':r['macro']['J_exact']} for r in trace}
        ca=auc(cr,'J_exact')
        if p is None or s is None or p['J_exact'] is None or s['J_exact'] is None or ca is None:continue
        num=p['J_exact']-s['J_exact'];den=ca-s['J_exact']
        calibration.append({'event_id':e,'numerator':num,'denominator':den,'ratio':num/den if abs(den)>=1e-8 else None})
    payload['behavior_oracle_opportunity']=calibration
    if calibration:
        n=float(np.mean([r['numerator'] for r in calibration]));d=float(np.mean([r['denominator'] for r in calibration]))
        payload['behavior_ratio_of_paired_means']={'numerator':n,'denominator':d,'ratio':n/d if abs(d)>=1e-8 else None}
    payload['signal_oracle_opportunity']=[]
    for e in events:
        p=e['arms']['FRESH_PDR'];s=e['arms']['STANDARD_TD']
        if p and s and p['weight_error'] is not None and s['weight_error'] is not None:
            num=s['weight_error']-p['weight_error'];den=s['weight_error']
            payload['signal_oracle_opportunity'].append({'event_id':e['event_id'],'numerator':num,'denominator':den,'ratio':num/den if abs(den)>=1e-8 else None,'CORRECT_H_weight_error':0.,'reason':'Exact correction equals exact-current weight on all H rows by construction during the early window.'})
    payload['adoption_at_risk']={}
    for arm in ('FRESH_PDR','STANDARD_TD'):
        obs=[e['arms'][arm] for e in events if e['arms'][arm] is not None]
        payload['adoption_at_risk'][arm]=[{'cycle':u,'at_risk':sum(v['adoption_time'] is None or v['adoption_time']+16>=u for v in obs),'confirmed_adoptions':sum(v['adoption_time'] is not None and v['adoption_time']+16<=u for v in obs),'right_censored_at_cap':sum(v['adoption_censored'] for v in obs) if u==2048 else 0} for u in range(0,2049,16)]
    payload['compute']=[]
    for e in events:
        run=root/'runs'/e['event_id']/'FRESH_PDR'
        if not (run/'pdr_compute_log.jsonl').exists():continue
        rows=[json.loads(l) for l in (run/'pdr_compute_log.jsonl').read_text().splitlines()]
        payload['compute'].append({'event_id':e['event_id'],'cycles':len(rows),'pdr_refreshes':sum(r['effective_arm']=='FRESH_PDR' for r in rows),'environment_transitions':sum(1 for _ in (run/'continuation_stream.jsonl').open()),'timing_totals':{k:sum(r[k] for r in rows) for k in ('policy_seconds','solve_seconds','lookup_seconds','actor_update_seconds','total_cycle_seconds')},'peak_memory_bytes':max(r['peak_memory_bytes'] for r in rows)})
    if out.exists(): raise RuntimeError('ANALYSIS_ALREADY_EXISTS')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(clean(payload),indent=2,allow_nan=False)+'\n')
    (root/'results/SC_D_FINAL_REPORT.md').write_text('# SC-D amended analysis\n\n'+payload['verdict']+'\n\nAll 50 parent events were retained. Event 000 had a disclosed one-update prelock engineering exposure with a mismatched pilot reference; the additional 49-event sensitivity is reported in the JSON. This is not a completely unexposed 50-event preregistration.\n')

if __name__=='__main__':main()
