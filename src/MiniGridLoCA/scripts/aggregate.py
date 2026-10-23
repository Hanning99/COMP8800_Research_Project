"""Regenerate neural-reference statistics and figures from archived per-run CSVs."""
from pathlib import Path
import argparse,hashlib,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/minigridloca-release-mpl')
import numpy as np,pandas as pd
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
METRIC='task_b_exact_discounted_return'

def plot(out,stats):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'axes.linewidth':.7,'savefig.dpi':300})
    handles=[Line2D([],[],color=c,ls=s,lw=2 if m=='PDR' else 1.25,label=l) for m,l,c,s in zip(METHODS,LABELS,COLORS,STYLES)]
    with PdfPages(out/'MiniGridLoCA_neural_all_baselines.pdf') as pdf:
        for limit,stem in [(1024,'01_minigridloca_neural_early'),(4096,'02_minigridloca_neural_full')]:
            fig,axes=plt.subplots(1,3,figsize=(7.25,2.55),sharey=True)
            fig.subplots_adjust(left=.08,right=.99,bottom=.2,top=.73,wspace=.16)
            fig.legend(handles=handles,ncol=8,loc='upper center',bbox_to_anchor=(.52,.98),frameon=False,
                fontsize=7.4,columnspacing=1.,handlelength=1.8,handletextpad=.45)
            for ax,bb in zip(axes,['AWAC','SAC','QAC']):
                for m,c,s in zip(METHODS,COLORS,STYLES):
                    d=stats[(stats.backbone==bb)&(stats.method==m)&(stats['update']<=limit)].sort_values('update')
                    ax.fill_between(d['update'],d.ci_low,d.ci_high,color=c,alpha=.14 if m=='PDR' else .09,lw=0)
                    ax.plot(d['update'],d['mean'],color=c,ls=s,lw=2 if m=='PDR' else 1.25,zorder=6 if m=='PDR' else 3)
                ax.set_title(bb);ax.set_xlim(0,limit);ax.set_ylim(0,1.96);ax.set_xticks([0,limit//2,limit])
                ax.set_yticks([0,.5,1,1.5]);ax.grid(axis='y',color='#E5EAEE',lw=.65);ax.set_xlabel('Actor updates')
            axes[0].set_ylabel('Task-B discounted return')
            for ext in ['png','pdf','svg']:fig.savefig(out/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.03)
            pdf.savefig(fig,bbox_inches='tight',pad_inches=.03);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a new output directory')
    a.out.mkdir(parents=True)
    runs=json.loads((ROOT/'RUNS.json').read_text());draws=np.random.default_rng(20260925).integers(20,size=(10000,20))
    original=pd.read_csv(ROOT/'provenance/analysis/CURVE_STATISTICS.csv')
    stats=[];per=[];summary=[];paired=[];maxerror=0
    for bb in ['AWAC','SAC','QAC']:
        scores={}
        for method in METHODS:
            sub=[r for r in runs if r['backbone']==bb and r['method']==method]
            assert len(sub)==20 and len({r['seed'] for r in sub})==20
            ys=[];es=[];fs=[];grid=None
            for r in sub:
                path=ROOT/r['curve'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['curve_sha256']
                df=pd.read_csv(path);x=df['update'].to_numpy();y=df[METRIC].to_numpy();assert np.isfinite(y).all()
                if grid is None:grid=x
                assert np.array_equal(grid,x) and x[0]==0 and x[-1]==4096
                early=x<=1024;e=float(np.trapezoid(y[early],x[early])/1024);f=float(np.trapezoid(y,x)/4096)
                assert abs(e-r['early_mean'])<1e-12 and abs(y[-1]-r['final_return'])<1e-12
                per.append(dict(backbone=bb,method=method,seed=r['seed'],early_mean=e,full_mean=f,final=y[-1],curve=r['curve']))
                ys.append(y);es.append(e);fs.append(f)
            ys=np.array(ys);es=np.array(es);scores[method]=es
            mean=ys.mean(0);lo,hi=np.quantile(ys[draws].mean(1),[.025,.975],axis=0)
            old=original[(original.view=='mapped_best')&(original.cell==sub[0]['cell']['id'])&(original.method==method)].sort_values('update')
            assert np.array_equal(old['update'],grid)
            error=float(np.max(np.abs(old[['mean','ci_low','ci_high']].to_numpy()-np.stack([mean,lo,hi],axis=1))))
            assert error<1e-12;maxerror=max(maxerror,error)
            for u,m,l,h in zip(grid,mean,lo,hi):stats.append(dict(backbone=bb,method=method,n=20,update=int(u),mean=m,ci_low=l,ci_high=h))
            ci=np.quantile(es[draws].mean(1),[.025,.975]);fci=np.quantile(ys[:,-1][draws].mean(1),[.025,.975])
            summary.append(dict(backbone=bb,method=method,n=20,early_mean=es.mean(),early_ci_low=ci[0],early_ci_high=ci[1],
                full_mean=np.mean(fs),final_mean=mean[-1],final_ci_low=fci[0],final_ci_high=fci[1]))
        for method in METHODS[:-1]:
            d=scores['PDR']-scores[method];ci=np.quantile(d[draws].mean(1),[.025,.975])
            paired.append(dict(backbone=bb,baseline=method,early_mean_difference=d.mean(),ci_low=ci[0],ci_high=ci[1],positive_pairs=int((d>0).sum())))
    stats=pd.DataFrame(stats)
    for name,data in [('CURVE_STATISTICS',stats),('PER_SEED',per),('SUMMARY',summary),('PAIRED_COMPARISONS',paired)]:
        pd.DataFrame(data).to_csv(a.out/f'{name}.csv',index=False)
    plot(a.out,stats)
    report=dict(status='PASS',runs=len(runs),reference='neural',view='mapped_best',paired_seeds_per_backbone=20,
        maximum_error_against_original_plotted_means_and_CIs=maxerror,bootstrap_seed=20260925,bootstrap_draws=10000,
        early_window=1024,full_window=4096,scope='Recomputed from raw CSVs; no smoothing, no run exclusion, no retraining.')
    (a.out/'REBUILD_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
