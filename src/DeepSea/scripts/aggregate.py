"""Recompute archived bootstrap statistics from raw curves, then draw figures."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/deepsea_release_mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
R=Path(__file__).resolve().parents[1]
METHODS=['TD','PER','DISCOR','EXTRA_TD','IQL','CQL','PS','OURS']
LABELS=['TD','PER','DisCor','8× TD','IQL','CQL','PS + actor','PDR-SAC']
COLORS=['#68737D','#A398B7','#A47551','#70A3BF','#D39017','#2564A7','#008979','#BE2637']
STYLES=[(0,(4,2)),(0,(1,1.6)),(0,(3,1.8)),(0,(4,1.5,1,1.5)),'-','-','-','-']
SIZES=[10,20,30,40]
def plot(out,curves):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
        'xtick.labelsize':7.2,'ytick.labelsize':7.2,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.6,'axes.edgecolor':'#5C6670','xtick.major.width':.6,'ytick.major.width':.6,
        'xtick.major.size':3,'ytick.major.size':3,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'figure.facecolor':'white','savefig.facecolor':'white','savefig.dpi':320,
        'text.color':'#1A2533','axes.labelcolor':'#1A2533','xtick.color':'#3D4956','ytick.color':'#3D4956',
        'mathtext.fontset':'dejavusans'})
    handles=[Line2D([],[],color=c,ls=s,lw=1.65 if m=='OURS' else 1.15,label=l) for m,l,c,s in zip(METHODS,LABELS,COLORS,STYLES)]
    def draw(ax,n,xmax):
        for m,c,s in zip(METHODS,COLORS,STYLES):
            x,y,lo,hi=curves[n,m];keep=x<=xmax;z=6 if m=='OURS' else (4 if m=='CQL' else 2)
            ax.fill_between(x[keep],lo[keep],hi[keep],color=c,alpha=.13,lw=0,zorder=z-1)
            ax.plot(x[keep],y[keep],color=c,ls=s,lw=1.7 if m=='OURS' else (1.4 if m=='CQL' else 1.05),zorder=z)
        ax.set_xlim(0,xmax);ax.set_ylim(-.025,1.035);ax.set_yticks([0,.25,.5,.75,1]);ax.set_xticks([0,xmax//2,xmax])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:str(int(x))))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:g}'))
        ax.grid(axis='y',color='#E3E8ED',lw=.55,zorder=0);ax.set_axisbelow(True)
    def legend(fig):
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.985),ncol=8,frameon=False,
            fontsize=7.1,columnspacing=1.,handlelength=1.8,handletextpad=.45,borderaxespad=0)
    def save(fig,name,pdf):
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}')
        pdf.savefig(fig);plt.close(fig)
    with PdfPages(out/'DeepSea_N10_N20_N30_N40.pdf') as pdf:
        fig,axes=plt.subplots(2,4,figsize=(7.25,4.),sharey=True)
        fig.subplots_adjust(left=.088,right=.963,bottom=.135,top=.845,wspace=.20,hspace=.38);legend(fig)
        for col,n in enumerate(SIZES):
            for row,xmax in enumerate([256,2048]):
                draw(axes[row,col],n,xmax)
                if row==0:axes[row,col].set_title(f'$N={n}$',pad=7)
                if col==0:axes[row,col].set_ylabel('Expected return',labelpad=6)
                if row==1:axes[row,col].set_xlabel('Actor updates',labelpad=4)
        save(fig,'01_deepsea_all_baselines_early',pdf)
        fig,axes=plt.subplots(1,4,figsize=(7.25,2.35),sharey=True)
        fig.subplots_adjust(left=.088,right=.963,bottom=.215,top=.735,wspace=.20);legend(fig)
        for ax,n in zip(axes,SIZES):draw(ax,n,8192);ax.set_title(f'$N={n}$',pad=7);ax.set_xlabel('Actor updates',labelpad=4)
        axes[0].set_ylabel('Expected return',labelpad=6);save(fig,'02_deepsea_all_baselines_full',pdf)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists(),'Output must be new';a.out.mkdir(parents=True)
    records=json.loads((R/'RUNS.json').read_text());curves={};raw=[];stats=[];summary=[];paired=[];counts={}
    frozen=R/'provenance/frozen_analysis';oldper=pd.read_csv(frozen/'PER_SEED.csv').set_index(['N','method','seed'])
    maxscore=0.
    for n in SIZES:
        seeds=[r['seed'] for r in records if r['N']==n and r['method']=='OURS'];counts[n]=len(seeds)
        idx=np.random.default_rng(20260925).integers(len(seeds),size=(10000,len(seeds)));values={}
        for m in METHODS:
            runs={r['seed']:r for r in records if r['N']==n and r['method']==m};assert set(runs)==set(seeds)
            ys=[];scores=[];x=None
            for seed in seeds:
                r=runs[seed];d=pd.read_csv(R/r['curve']);xx=d['update'].to_numpy();y=d['expected_return'].to_numpy()
                assert xx[0]==0 and xx[-1]==8192 and np.isfinite(y).all() and (np.diff(xx)>0).all()
                if x is None:x=xx
                else:assert np.array_equal(xx,x)
                early=np.trapezoid(y[x<=2048],x[x<=2048])/2048;full=np.trapezoid(y,x)/8192
                old=oldper.loc[(n,m,seed)];maxscore=max(maxscore,abs(early-old.early),abs(y[-1]-old.final))
                raw.append(dict(N=n,method=m,seed=seed,early=early,full=full,final=y[-1],curve=r['curve']))
                ys.append(y);scores.append([early,full,y[-1]])
            Y=np.asarray(ys);v=np.asarray(scores);values[m]=v;mean=Y.mean(0);lo,hi=np.quantile(Y[idx].mean(1),[.025,.975],axis=0)
            curves[n,m]=(x,mean,lo,hi)
            for j,t in enumerate(x):stats.append(dict(N=n,method=m,n=len(seeds),update=int(t),mean=mean[j],ci_low=lo[j],ci_high=hi[j]))
            row=dict(N=n,method=m,n=len(seeds),recipe=r['recipe']['id'])
            if r['condition']:row['condition']=r['condition']['id']
            for j,name in enumerate(['early','full','final']):
                ci=np.quantile(v[idx,j].mean(1),[.025,.975]);row.update({name+'_mean':v[:,j].mean(),name+'_ci_low':ci[0],name+'_ci_high':ci[1]})
            summary.append(row)
        for m in METHODS[:-1]:
            for j,name in enumerate(['early','full','final']):
                gain=values['OURS'][:,j]-values[m][:,j];ci=np.quantile(gain[idx].mean(1),[.025,.975])
                paired.append(dict(N=n,baseline=m,metric=name,n=len(seeds),gain=gain.mean(),ci_low=ci[0],ci_high=ci[1]))
    st=pd.DataFrame(stats);su=pd.DataFrame(summary);oldst=pd.read_csv(frozen/'CURVE_STATISTICS.csv');oldsu=pd.read_csv(frozen/'SUMMARY.csv')
    errors={}
    for current,old,keys,cols in [(st,oldst,['N','method','update'],['mean','ci_low','ci_high']),(su,oldsu,['N','method'],['early_mean','early_ci_low','early_ci_high','final_mean'])]:
        merged=current.merge(old,on=keys,suffixes=('_new','_old'));assert len(merged)==len(current)==len(old)
        for col in cols:errors['_'.join(keys)+'/'+col]=float(np.max(np.abs(merged[col+'_new']-merged[col+'_old'])))
    assert max(errors.values())<1e-12 and maxscore<1e-12,(errors,maxscore)
    for name,frame in [('PER_SEED',pd.DataFrame(raw)),('CURVE_STATISTICS',st),('SUMMARY',su),('PAIRED_COMPARISONS',pd.DataFrame(paired))]:frame.to_csv(a.out/f'{name}.csv',index=False)
    configs=[]
    for n in SIZES:
        for m in METHODS:
            group=[r for r in records if r['N']==n and r['method']==m]
            assert len({json.dumps([r['recipe'],r['condition']],sort_keys=True) for r in group})==1
            r=group[0];configs.append(dict(N=n,method=m,n=len(group),actor='SAC' if m in ['CQL','OURS'] else ('IQL' if m=='IQL' else 'AWAC'),recipe=json.dumps(r['recipe'],sort_keys=True),condition=json.dumps(r['condition'],sort_keys=True)))
    pd.DataFrame(configs).to_csv(a.out/'CONFIGURATIONS.csv',index=False)
    plot(a.out,curves)
    report=dict(status='PASS',raw_curves=len(records),paired_seeds=counts,max_score_error=maxscore,statistics_max_errors=errors,
        bootstrap_draws=10000,bootstrap_seed=20260925,early='trapezoidal mean return, updates 0–2048',full='trapezoidal mean return, updates 0–8192',final='expected return at update 8192')
    (a.out/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
