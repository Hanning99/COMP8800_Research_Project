"""Run small end-to-end checks when mounted at /release by an isolated runtime."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    forbidden=['/data3/hanning/Reproduce','/data3/hanning/autoresearch/SecureCourier-Mechanism-20260902']
    assert not any(Path(p).exists() for p in forbidden),'Original sources must not be mounted'
    commands=[
        ['scripts/export_csv.py'],
        ['scripts/plot.py'],
        ['scripts/rescore.py','--events','confirmatory_000','confirmatory_025','confirmatory_049',
         '--cycles','0','1024','2048','--workers','2','--native','--output','validation/native_isolated.csv'],
        ['scripts/check_method.py','--output','validation/method_isolated.json'],
    ]
    executed=[]
    for args in commands:
        subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
        executed.append(args)
    import torch,numpy,matplotlib
    report=dict(status='PASS',package_mount=str(ROOT),original_sources_unmounted=forbidden,
        commands=executed,python=platform.python_version(),torch=torch.__version__,
        numpy=numpy.__version__,matplotlib=matplotlib.__version__,cuda_used=False)
    (ROOT/'validation/isolation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
