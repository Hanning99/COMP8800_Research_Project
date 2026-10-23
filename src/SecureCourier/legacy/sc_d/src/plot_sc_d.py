"""Standalone paper artifacts from locked event-level results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parent.parent
    result=json.loads((root/'results/SC_D_LOCKED_ANALYSIS.json').read_text())
    out=root/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    grid=np.arange(0,1025,16)
    for arm,color in [('STANDARD_TD','#777777'),('FRESH_PDR','#0072B2')]:
        traces={key:[] for key in ['weight_error','log_odds','J_exact']}
        for e in result['events']:
            path=root/'scored'/e['event_id']/arm
            if not (path/'COMPLETE.json').exists():continue
            values=[json.loads((path/f'cycle_{u:06d}.json').read_text()) for u in grid]
            for k in traces:traces[k].append([r[k] for r in values])
        for ax,(key,data) in zip(axes,traces.items()):
            if not data:continue
            array=np.asarray(data,dtype=float)
            ax.plot(grid,np.nanmean(array,axis=0),label=arm,color=color)
            ax.set(xlabel='Learner cycle',ylabel=key)
    axes[0].legend(frameon=False)
    fig.savefig(out/'SC_D_main.pdf');fig.savefig(out/'SC_D_main.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
    for arm,color in [('STANDARD_TD','#777777'),('FRESH_PDR','#0072B2')]:
        observations=[e['arms'][arm] for e in result['events'] if e['arms'][arm] is not None]
        curve=[sum(v['adoption_time'] is not None and v['adoption_time']+16<=u for v in observations)/len(observations) for u in range(0,2049,16)] if observations else []
        if curve:axes[0].step(range(0,2049,16),curve,label=arm,color=color,where='post')
    axes[0].set(xlabel='Learner cycle',ylabel='Persistent adoption fraction');axes[0].legend(frameon=False)
    pairs=[(i,e['differences']['J_exact']) for i,e in enumerate(result['events']) if e['differences']['J_exact'] is not None]
    if pairs:axes[1].scatter([x for x,y in pairs],[y for x,y in pairs],s=12)
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(xlabel='Source event',ylabel='PDR − Standard EarlyAUC')
    fig.savefig(out/'SC_D_event_adoption.pdf');plt.close(fig)
    (out/'caption.md').write_text('SecureCourier SC-D, original 50 source-event cohort; surfaces are macro-averaged within event. PDR uses a frozen pre-capability actor and learned single critic on a simulator-derived closed finite-horizon graph, refreshed before each actor update during cycles 1–1024. Both arms train the ordinary critic. Exact-current scoring is post-training only. One event had disclosed prelock engineering exposure; the 49-event sensitivity accompanies the full-cohort analysis. Curves show event means; inferential intervals are paired event bootstrap intervals in the result table. Learner-cycle differences do not establish compute efficiency.\n')

if __name__=='__main__':main()
