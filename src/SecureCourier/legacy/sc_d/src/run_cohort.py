"""Bounded GPU queue; persist exit status, never select events by performance."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from scd_common import PARENT, write_pretty_json
from lock_sc_d import verify_lock

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpus',nargs='+',required=True)
    parser.add_argument('--per-gpu',type=int,default=2)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    verify_lock(root)
    logroot=root/'logs';logroot.mkdir(exist_ok=True)
    queue=[f'confirmatory_{i:03d}' for i in range(50)]
    slots=[(gpu,j) for gpu in args.gpus for j in range(args.per_gpu)]
    running={}; terminal={}
    while queue or running:
        for slot in slots:
            if slot in running or not queue: continue
            event=queue.pop(0)
            out=root/'runs'/event/'FRESH_PDR'
            cmd=[sys.executable,str(root/'src/run_sc_d.py'),'--source-event',str(PARENT/'source_events'/event),'--reference',str(root/'references'/event),'--graph',str(root/'manifests/support_graph.json'),'--arm','FRESH_PDR','--output',str(out),'--device','cuda:0','--mode','confirmatory','--max-cycles','2048','--checkpoint-interval','16','--evaluation-interval','16','--disable-scoring']
            if (out/'RUN_MANIFEST.json').exists():
                m=json.loads((out/'RUN_MANIFEST.json').read_text())
                if m['status']=='SEALED_COMPLETE':
                    terminal[event]={'phase':'training','exit_code':0,'existing_sealed':True};continue
            if out.exists():
                if not list((out/'checkpoints').glob('cycle_*')):
                    terminal[event]={'phase':'training','exit_code':-1,'reason':'existing output has no complete checkpoint'};continue
                cmd+=['--resume']
            log=(logroot/(event+'.log')).open('ab')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':slot[0],'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            running[slot]=(event,proc,log)
            print(f'START {event} gpu={slot[0]} pid={proc.pid}',flush=True)
        for slot,(event,proc,log) in list(running.items()):
            code=proc.poll()
            if code is None: continue
            log.close();del running[slot]
            terminal[event]={'phase':'training','exit_code':code}
            print(f'EXIT {event} code={code}',flush=True)
        write_pretty_json(logroot/'COHORT_STATUS.json',{'queued':len(queue),'running':{e:{'pid':p.pid,'gpu':slot[0]} for slot,(e,p,_) in running.items()},'terminal':terminal})
        if queue or running:time.sleep(5)
    write_pretty_json(logroot/'TRAINING_TERMINAL.json',{'events':terminal})
    # Scoring has its own GPU queue; it cannot communicate values to training.
    jobs=[(e,a) for e in sorted(terminal) for a in ['STANDARD_TD','FRESH_PDR'] if a=='STANDARD_TD' or terminal[e]['exit_code']==0]
    running={};score_status={}
    while jobs or running:
        for slot in slots:
            if slot in running or not jobs:continue
            event,arm=jobs.pop(0)
            log=(logroot/(event+'_'+arm+'_scoring.log')).open('ab')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':slot[0],'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
            proc=subprocess.Popen([sys.executable,str(root/'src/scoring_sc_d.py'),'--event',event,'--arm',arm,'--device','cuda:0'],env=env,stdout=log,stderr=subprocess.STDOUT)
            running[slot]=(event,arm,proc,log)
        for slot,(event,arm,proc,log) in list(running.items()):
            code=proc.poll()
            if code is None:continue
            log.close();del running[slot];score_status[event+'/'+arm]=code
        write_pretty_json(logroot/'SCORING_STATUS.json',{'queued':len(jobs),'running':len(running),'terminal':score_status})
        if jobs or running:time.sleep(5)
    write_pretty_json(logroot/'COHORT_TERMINAL.json',{'events':terminal,'scoring':score_status})
    subprocess.run([sys.executable,str(root/'src/analyze_sc_d.py')],check=True)
    subprocess.run([sys.executable,str(root/'src/plot_sc_d.py')],check=True)

if __name__=='__main__':main()
