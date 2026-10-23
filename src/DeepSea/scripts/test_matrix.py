"""Limited GPU validation, not a claim that all 520 trajectories were retrained."""
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,sys
R=Path(__file__).resolve().parents[1]
METHODS=['TD','PER','DISCOR','EXTRA_TD','IQL','CQL','PS','PDR']
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1');p.add_argument('--out',type=Path,required=True);p.add_argument('--full-pdr',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);gpus=a.gpus.split(',')
    def work(index,gpu):
        reports=[]
        for n in [10,20,30,40][index::len(gpus)]:
            jobs=[(m,mode,16) for m in METHODS for mode in ['replay','final']]
            if a.full_pdr:jobs.append(('PDR','full',8192))
            for method,mode,updates in jobs:
                name=f'N{n}_{method}_{mode}';out=a.out/name
                if not (out/'VERIFICATION.json').exists():
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PDR_DEVICE='cuda:0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
                    cmd=[sys.executable,str(R/'scripts/reproduce.py'),'--size',str(n),'--method',method,'--mode','replay' if mode=='full' else mode,'--updates',str(updates),'--out',str(out)]
                    with (a.out/f'{name}.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,cwd='/tmp')
                z=json.loads((out/'VERIFICATION.json').read_text());assert z['status']=='PASS';reports.append(z)
                print('PASS',name,flush=True)
        return reports
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        groups=list(pool.map(lambda z:work(*z),enumerate(gpus)))
    reports=[r for group in groups for r in group];expected=68 if a.full_pdr else 64;assert len(reports)==expected
    (a.out/'MATRIX_VERIFICATION.json').write_text(json.dumps(dict(status='PASS',checks=len(reports),reports=reports),indent=2)+'\n')
if __name__=='__main__':main()
