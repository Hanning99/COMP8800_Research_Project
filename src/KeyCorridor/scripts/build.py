from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
(root/'build').mkdir(exist_ok=True)
subprocess.run(['g++','-O3','-std=c++17','-ffp-contract=off','-shared','-fPIC',str(root/'src/planning.cpp'),'-o',str(root/'build/planning.so')],check=True)
print('Built prioritized-sweeping kernel with archived compiler flags.')
