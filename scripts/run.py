"""One entry point for checkpoint evaluation, result reconstruction and replay."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['evaluate','reproduce','results'])
 p.add_argument('--benchmark',choices=['DeepSea','KeyCorridor','MiniGridLoCA','SecureCourier'],default='DeepSea')
 p.add_argument('--archive',type=Path,default=Path(os.environ.get('PDR_CHECKPOINT_ARCHIVE',str(ROOT/'checkpoints/local'))))
 p.add_argument('--data-root',type=Path,help='Already restored release root containing RUNS.json')
 p.add_argument('--workspace',type=Path,default=ROOT/'work')
 p.add_argument('--out',type=Path);p.add_argument('--limit',type=int,help='Evaluation subset only; omit to evaluate all final checkpoints')
 p.add_argument('--representative',action='store_true',help='One seed per task/size/backbone/method; evaluation subset')
 a,extra=p.parse_known_args()
 if extra and extra[0]=='--':extra=extra[1:]
 if a.action!='reproduce' and a.benchmark!='SecureCourier' and extra:p.error('Unexpected arguments: '+' '.join(extra))
 stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
 output=(a.out or ROOT/'outputs'/f'{a.benchmark}-{a.action}-{stamp}').resolve()
 if output.exists():raise FileExistsError('Output must be new: '+str(output))
 if a.data_root:
  data=a.data_root.resolve()
 else:
  archive=a.archive.resolve()
  if not (archive/'manifest.sqlite').exists():raise FileNotFoundError('Checkpoint archive not available. Follow checkpoints/README.md or pass --archive /path/to/archive.')
  a.workspace.mkdir(parents=True,exist_ok=True);restored=(a.workspace/f'{a.benchmark}-{stamp}').resolve()
  subprocess.run([sys.executable,str(ROOT/'scripts/checkpoint_archive.py'),'restore','--archive-root',str(archive),'--release',a.benchmark,'--output',str(restored)],check=True)
  data=restored/'AAMAS-Experiment/Effectiveness'/a.benchmark
 if a.benchmark!='SecureCourier' and not (data/'RUNS.json').is_file():raise FileNotFoundError(data/'RUNS.json')
 if a.benchmark=='SecureCourier' and not (data/'provenance/cohort.json').is_file():raise FileNotFoundError(data/'provenance/cohort.json')
 # Check that the numerical code distributed in GitHub matches the restored release.
 manifest=json.loads((ROOT/'configs/PRIMARY_SOURCE_FILES.json').read_text())
 import hashlib
 for r in manifest:
  prefix='src/releases/'+a.benchmark+'/'
  if not r['path'].startswith(prefix):continue
  rel=r['path'][len(prefix):]
  path=data/rel
  if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise RuntimeError('Restored release code mismatch: '+rel)
 if a.benchmark=='SecureCourier':
  if a.action=='evaluate':cmd=[sys.executable,str(data/'scripts/rescore.py'),'--output',str(output/'EVALUATION.csv'),*extra]
  elif a.action=='results':cmd=[sys.executable,str(data/'scripts/export_csv.py'),'--output',str(output)]
  else:cmd=[sys.executable,str(data/'scripts/train.py'),'--output',str(output),*extra]
 elif a.action=='evaluate':
  cmd=[sys.executable,str(ROOT/'scripts/evaluate_cpu.py'),'--benchmark',a.benchmark,'--data-root',str(data),'--out',str(output/'EVALUATION.json')]
  if a.limit is not None:cmd+=['--limit',str(a.limit)]
  if a.representative:cmd+=['--representative']
 elif a.action=='results':cmd=[sys.executable,str(data/'scripts/aggregate.py'),'--out',str(output)]
 else:cmd=[sys.executable,str(data/'scripts/reproduce.py'),'--out',str(output),*extra]
 env=dict(os.environ);env.setdefault('OMP_NUM_THREADS','1');env.setdefault('OPENBLAS_NUM_THREADS','1')
 subprocess.run(cmd,cwd=data,env=env,check=True)
 print('Completed:',output)
if __name__=='__main__':main()
