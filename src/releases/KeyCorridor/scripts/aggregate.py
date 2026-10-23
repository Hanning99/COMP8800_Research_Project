"""Raw per-seed CSV -> statistics -> neural-only figures. No training/imports of Torch."""
from pathlib import Path
import argparse, hashlib, json, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/keycorridor-release-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
METHODS=['TD','PER','DISCOR','EXTRA_TD','IQL','CQL','PS','PDR']
LABELS=['TD','PER','DisCor','8× TD','IQL','CQL','PS + actor','PDR']
COLORS=['#68737D','#A398B7','#A47551','#70A3BF','#D39017','#2564A7','#008979','#BE2637']
STYLES=[(0,(5,2)),(0,(1,2)),(0,(3,2)),(0,(5,2,1,2)),'-','-','-','-']
TASKS=['S3R2','S3R3','S4R3','S5R3']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError('Use a new or empty output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    runs=json.loads((ROOT/'RUNS.json').read_text());assert len(runs)==640
    per=[];stats=[];summary=[];paired=[];settings=[]
    for task in TASKS:
        cell=f'KeyCorridor_{task}_neural_AWAC';scores={};seed_order=None
        draws=np.random.default_rng(20260925).integers(20,size=(10000,20))
        for method in METHODS:
            group=[r for r in runs if r['cell']['id']==cell and r['method']==method]
            seeds=[r['seed'] for r in group];assert len(set(seeds))==20
            if seed_order is None:seed_order=seeds
            assert seed_order==seeds,'Paired bootstrap requires identical seed ordering'
            ys=[];es=[];fs=[];ends=[]
            for r in group:
                path=ROOT/r['curve'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['curve_sha256']
                data=pd.read_csv(path);x=data['update'].to_numpy();y=data.success.to_numpy()
                assert x[0]==0 and x[-1]==512 and np.all(np.diff(x)>0) and np.isfinite(y).all()
                assert np.all((y>=0)&(y<=1));mask=x<=128;assert x[mask][-1]==128
                if ys:assert np.array_equal(x,x0)
                x0=x;early=np.trapezoid(y[mask],x[mask])/128;full=np.trapezoid(y,x)/512
                ys.append(y);es.append(early);fs.append(full);ends.append(y[-1])
                per.append(dict(cell=cell,task=task,method=method,recipe=r['config']['id'],seed=r['seed'],
                    early_mean=early,full_mean=full,final=y[-1],curve=r['curve'],sha256=r['curve_sha256']))
            y=np.asarray(ys);mean=y.mean(0);lo,hi=np.quantile(y[draws].mean(1),[.025,.975],axis=0)
            scores[method]=dict(early=np.asarray(es),full=np.asarray(fs),final=np.asarray(ends))
            for j,u in enumerate(x):stats.append(dict(cell=cell,task=task,method=method,recipe=r['config']['id'],update=int(u),n=20,mean=mean[j],ci_low=lo[j],ci_high=hi[j]))
            s=dict(cell=cell,task=task,method=method,recipe=r['config']['id'],n=20)
            for metric,values in scores[method].items():
                low,high=np.quantile(values[draws].mean(1),[.025,.975])
                s.update({metric+'_mean':values.mean(),metric+'_ci_low':low,metric+'_ci_high':high})
            summary.append(s);settings.append({'cell':cell,'method':method,**r['config']})
        for method in METHODS[:-1]:
            for metric in ['early','full','final']:
                difference=scores['PDR'][metric]-scores[method][metric]
                lo,hi=np.quantile(difference[draws].mean(1),[.025,.975])
                paired.append(dict(cell=cell,baseline=method,metric=metric,n=20,gain=difference.mean(),ci_low=lo,ci_high=hi))
    generated=pd.DataFrame(stats)
    archived=pd.read_csv(ROOT/'provenance/frozen_analysis/CURVE_STATISTICS.csv')
    merged=generated.merge(archived,on=['cell','method','update'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(merged)==len(generated)==len(archived)
    errors={key:float(np.max(np.abs(merged[key+'_new']-merged[key+'_old']))) for key in ['mean','ci_low','ci_high']}
    assert max(errors.values())<1e-12,errors
    for name,rows in [('PER_SEED',per),('CURVE_STATISTICS',stats),('SUMMARY',summary),('PAIRED_COMPARISONS',paired),('CONFIGURATIONS',settings)]:
        pd.DataFrame(rows).to_csv(a.out/f'{name}.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'pdf.fonttype':42})
    handles=[Line2D([],[],color=c,ls=s,lw=2.7 if m=='PDR' else 1.8,label=l) for m,c,s,l in zip(METHODS,COLORS,STYLES,LABELS)]
    with PdfPages(a.out/'KeyCorridor_neural_all_baselines.pdf') as pdf:
        for limit,name in [(128,'01_keycorridor_neural_early'),(512,'02_keycorridor_neural_full')]:
            fig,axes=plt.subplots(1,4,figsize=(12,3.4));fig.subplots_adjust(left=.05,right=.99,bottom=.17,top=.76,wspace=.3)
            fig.legend(handles=handles,ncol=8,loc='upper center',bbox_to_anchor=(.52,.99),frameon=False,columnspacing=1.2)
            for ax,task in zip(axes,TASKS):
                for method,color,style in zip(METHODS,COLORS,STYLES):
                    d=generated[(generated.task==task)&(generated.method==method)&(generated['update']<=limit)]
                    ax.fill_between(d['update'],100*d.ci_low,100*d.ci_high,color=color,alpha=.14 if method=='PDR' else .08,lw=0)
                    ax.plot(d['update'],100*d['mean'],color=color,ls=style,lw=2.7 if method=='PDR' else 1.7,zorder=6 if method=='PDR' else 3)
                ax.set(title=task,xlabel='Actor updates',xlim=(0,limit),ylim=(0,100))
                ax.set_xticks(np.linspace(0,limit,5));ax.grid(axis='y',color='#E5EAEE',lw=.7)
            axes[0].set_ylabel('Task success (%)')
            for ext in ['pdf','png','svg']:fig.savefig(a.out/f'{name}.{ext}')
            pdf.savefig(fig);plt.close(fig)
    report=dict(status='PASS',raw_curves=640,tasks=4,methods=8,seeds_per_setting=20,
        archived_statistics_max_abs_error=errors,bootstrap=dict(seed=20260925,resamples=10000,type='paired seed percentile'),
        intervals='Across-seed variation, not corrected for development selection',early='trapezoidal AUC [0,128]/128',full='trapezoidal AUC [0,512]/512',final='success at update 512')
    (a.out/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
