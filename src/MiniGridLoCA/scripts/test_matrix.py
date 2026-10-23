"""Bounded replay/evaluation validation, one process per explicitly selected GPU."""
from pathlib import Path
import argparse,concurrent.futures,json,os,queue,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
METHODS=['TD','PER','DISCOR','EXTRA_TD','IQL','CQL','PS','PDR']

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    jobs=queue.Queue();answers=[]
    for bb in ['AWAC','SAC','QAC']:
        for method in METHODS:
            for mode in ['short','final']:jobs.put((bb,method,mode))
        for method in ['PDR','CQL','IQL']:jobs.put((bb,method,'full'))
    def work(gpu):
        while True:
            try:bb,method,kind=jobs.get_nowait()
            except queue.Empty:return
            name=f'{bb}_{method}_{kind}';out=a.out/name
            cmd=[sys.executable,str(ROOT/'scripts/reproduce.py'),'--backbone',bb,'--method',method,'--out',str(out)]
            if kind=='short':cmd+=['--updates','32']
            if kind=='final':cmd+=['--mode','final']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONNOUSERSITE='1')
            t=time.perf_counter()
            with (a.out/f'{name}.log').open('w') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            row=dict(name=name,gpu=gpu,returncode=result.returncode,elapsed_seconds=time.perf_counter()-t)
            if (out/'VERIFICATION.json').exists():row['verification']=json.loads((out/'VERIFICATION.json').read_text())
            answers.append(row);print(json.dumps(row),flush=True);jobs.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus.split(','))) as pool:
        list(pool.map(work,a.gpus.split(',')))
    report=dict(status='PASS' if all(r['returncode']==0 for r in answers) else 'FAIL',tests=len(answers),
        short_replays=24,final_evaluations=24,full_4096_update_replays=9,details=answers)
    (a.out/'MATRIX.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))
    if report['status']!='PASS':sys.exit(1)

if __name__=='__main__':main()
