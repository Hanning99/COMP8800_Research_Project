"""Reconstruct final actors and compare exact metrics with archived raw curves."""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.dont_write_bytecode=True

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',choices=['DeepSea','KeyCorridor','MiniGridLoCA'],required=True)
 p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--limit',type=int);p.add_argument('--representative',action='store_true');p.add_argument('--tolerance',type=float,default=1e-5);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 import numpy as np,pandas as pd,torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 root=a.data_root.resolve();sys.path.insert(0,str(root/'vendor'))
 if a.benchmark=='DeepSea':
  import deepsea_experiment as core
  core.DEVICE=torch.device('cpu');graphs={}
  def evaluate(r,state):
   key=(r['N'],r['seed'])
   if key not in graphs:graphs[key]=dict(np.load(root/'inputs/deepsea/worlds'/f'N{key[0]}'/f'seed_{key[1]}'/'graph.npz'))
   g=graphs[key];actor=core.Actor(g['features'].shape[1]);actor.load_state_dict(state['models'][0]);policy=core.actor_policy(actor,g)
   return core.exact_metrics(policy,g)
 elif a.benchmark=='KeyCorridor':
  import key_exact as core
  core.DEVICE=torch.device('cpu');core.base.DEVICE=torch.device('cpu');graphs={}
  def evaluate(r,state):
   task=r['cell']['task']
   if task not in graphs:graphs[task]=core.load_graph(task)
   g=graphs[task];actor=core.base.Actor(g['features'].shape[1]);actor.load_state_dict(state['models'][0])
   policy=core.actor_policy(actor,torch.as_tensor(g['features']),torch.as_tensor(g['mask']))
   return core.exact_metrics(policy,g,core.TASKS[task][1])
 else:
  os.environ['MINIGRID_LOCA_PROTOCOL']=str(root/'inputs/loca_exact/PROTOCOL_LOCK.json')
  from run_study import Actor,extract_policy,GAMMA
  from minigrid_loca_core import build_graph,evaluate_policy
  g=build_graph();states=torch.arange(g.n_states)
  def evaluate(r,state):
   actor=Actor().eval();actor.load_state_dict(state['models'][0]);return evaluate_policy(g,extract_policy(actor,states),GAMMA)
 runs=json.loads((root/'RUNS.json').read_text());chosen=runs if a.limit is None else runs[:a.limit]
 if a.representative:
  if a.limit is not None:raise ValueError('Choose --limit or --representative')
  chosen=[];seen=set()
  for r in runs:
   cell=r.get('cell',{});key=(cell.get('task'),r.get('N'),r.get('backbone',cell.get('backbone')),r['method'])
   if key not in seen:chosen.append(r);seen.add(key)
 if not chosen:raise RuntimeError('No runs selected')
 reports=[]
 for i,r in enumerate(chosen):
  z=torch.load(root/r['final'],map_location='cpu',weights_only=False);metrics=evaluate(r,z.get('state',z))
  curve=pd.read_csv(root/r['curve']);final=curve.loc[curve['update'].idxmax()]
  errors={k:abs(float(v)-float(final[k])) for k,v in metrics.items() if k in final.index}
  if not errors:raise RuntimeError('No matching metric columns')
  checked_errors={k:v for k,v in errors.items() if a.benchmark!='DeepSea' or k=='expected_return'}
  if not checked_errors:raise RuntimeError('Primary evaluation metric missing')
  error=max(checked_errors.values())
  if error>a.tolerance:raise RuntimeError(f'Checkpoint/curve mismatch: run {i}, {errors}')
  reports.append({'seed':r['seed'],'method':r['method'],'checkpoint':r['final'],'update':int(final['update']),'metrics':{k:float(v) for k,v in metrics.items()},'errors':errors,'checked_errors':checked_errors})
  if a.representative or (i+1)%20==0:print(f'Final actors evaluated: {i+1}/{len(chosen)}',flush=True)
 report={'status':'PASS','benchmark':a.benchmark,'scope':'Final-checkpoint actor reconstruction and exact evaluation; no training rerun.','device':'CPU','selection':'first archived seed per task/size/backbone/method' if a.representative else ('prefix subset' if a.limit is not None else 'all runs'),'available_runs':len(runs),'evaluated_runs':len(reports),'tolerance':a.tolerance,'maximum_metric_error':max(max(r['checked_errors'].values()) for r in reports),'comparison_scope':'DeepSea expected_return; other benchmarks all shared exact metrics. Auxiliary DeepSea counts retained without a unit-probability absolute threshold.','maximum_auxiliary_error':max(max(r['errors'].values()) for r in reports),'runs':reports}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='runs'}))
if __name__=='__main__':main()
