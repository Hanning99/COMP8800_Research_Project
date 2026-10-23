"""Draw both panels solely from curve_summary.csv; edit this file for new styles."""
import argparse
import csv
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,default=ROOT/'data/curve_summary.csv')
    p.add_argument('--output',type=Path,default=ROOT/'figures')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with a.csv.open(newline='') as f: rows=list(csv.DictReader(f))
    style={'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'legend.fontsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'securecourier-eq1',
        'savefig.facecolor':'white','path.simplify':False}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(7.0,2.8),layout='constrained')
        for ax,metric,title,ylabel in zip(axes,['p_shortcut','J_exact'],
            ['(a) Shortcut choice at the junction','(b) Expected return from the depot'],
            ['Shortcut probability','Expected discounted return']):
            for arm,label,color,linestyle in [('STANDARD_TD','Standard TD','#4575B4','--'),('FRESH_PDR','PDR','#D73027','-')]:
                r=sorted([r for r in rows if r['metric']==metric and r['arm']==arm and int(r['cycle'])<=1024],key=lambda r:int(r['cycle']))
                x=np.array([int(v['cycle']) for v in r]);np.testing.assert_array_equal(x,np.arange(0,1025,16))
                y=np.array([float(v['mean']) for v in r]);lo=np.array([float(v['ci95_low']) for v in r]);hi=np.array([float(v['ci95_high']) for v in r])
                assert np.isfinite([y,lo,hi]).all() and np.all(lo<=y) and np.all(y<=hi)
                ax.fill_between(x,lo,hi,color=color,alpha=.14,linewidth=0)
                ax.plot(x,y,color=color,linestyle=linestyle,label=label,lw=1.6)
            ax.set_title(title,loc='left',pad=9)
            ax.set(xlabel='Learner cycles',ylabel=ylabel,xlim=(0,1024))
            ax.set_xticks([0,256,512,768,1024],['0','256','512','768','1,024'])
            ax.grid(axis='y',color='#E5E5E5',lw=.5);ax.set_axisbelow(True)
            ax.tick_params(width=.6,length=3)
        axes[0].set_ylim(0,.40);axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        axes[0].set_yticks([0,.1,.2,.3,.4]);axes[0].legend(frameon=False,loc='upper left')
        axes[1].yaxis.set_major_locator(MaxNLocator(5));axes[1].ticklabel_format(axis='y',style='plain',useOffset=False)
        for ext in ['pdf','svg','png']:
            metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else None)
            fig.savefig(a.output/f'securecourier_effectiveness.{ext}',dpi=300,metadata=metadata)
        plt.close(fig)
    print(f'Wrote PDF, SVG and PNG from {a.csv}')

if __name__=='__main__':main()
