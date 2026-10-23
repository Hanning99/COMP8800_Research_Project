"""Compile the archived prioritized-sweeping kernel from its C++ source."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    (ROOT/'build').mkdir(exist_ok=True)
    command=['g++','-O3','-std=c++17','-ffp-contract=off','-shared','-fPIC',str(ROOT/'src/planning.cpp'),'-o',str(ROOT/'build/planning.so')]
    subprocess.run(command,check=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(command=command,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        sha256=sha(ROOT/'build/planning.so'),original_sha256=sha(ROOT/'provenance/original_build/planning.so'))
    report['matches_original_binary']=report['sha256']==report['original_sha256']
    (ROOT/'build/BUILD.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
