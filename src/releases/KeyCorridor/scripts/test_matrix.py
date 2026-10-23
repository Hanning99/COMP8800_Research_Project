"""One representative seed for every task/method, plus four full PDR reruns."""
from pathlib import Path
import argparse,concurrent.futures,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--full-pdr',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    tasks=['S3R2','S3R3','S4R3','S5R3'];methods=['TD','PER','DISCOR','EXTRA_TD','IQL','CQL','PS','PDR'];gpus=a.gpus.split(',')
    def worker(task,gpu):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
        jobs=[('reference',[str(ROOT/'scripts/reference.py'),'--task',task])]
        for method in methods:
            for mode in ['replay','final']:
                jobs.append((method+'_'+mode,[str(ROOT/'scripts/reproduce.py'),'--task',task,'--method',method,'--mode',mode,'--updates','16']))
        if a.full_pdr:jobs.append(('PDR_full',[str(ROOT/'scripts/reproduce.py'),'--task',task,'--method','PDR','--updates','512']))
        reports=[]
        for name,cmd in jobs:
            out=a.out/(task+'_'+name)
            if (out/'VERIFICATION.json').exists():
                reports.append(json.loads((out/'VERIFICATION.json').read_text()));continue
            with (a.out/(task+'_'+name+'.log')).open('w') as log:
                result=subprocess.run([sys.executable,*cmd,'--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{task} {name} failed: inspect {log.name}')
            reports.append(json.loads((out/'VERIFICATION.json').read_text()))
            print(f'PASS {task} {name}',flush=True)
        return reports
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        fs=[pool.submit(worker,t,gpus[i%len(gpus)]) for i,t in enumerate(tasks)]
        reports=[r for f in fs for r in f.result()]
    (a.out/'MATRIX_VERIFICATION.json').write_text(json.dumps(dict(status='PASS',checks=len(reports),reports=reports),indent=2)+'\n')

if __name__=='__main__':main()
