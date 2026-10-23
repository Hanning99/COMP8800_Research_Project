from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parents[1]
def main():
    (R/'build').mkdir(exist_ok=True);records=[]
    for name in ['kernels','planning']:
        cmd=['g++','-O3','-std=c++17','-ffp-contract=off','-shared','-fPIC',str(R/'src'/f'{name}.cpp'),'-o',str(R/'build'/f'{name}.so')]
        subprocess.run(cmd,check=True)
        records.append(dict(command=cmd,sha256=hashlib.sha256((R/'build'/f'{name}.so').read_bytes()).hexdigest()))
    (R/'build/BUILD.json').write_text(json.dumps(records,indent=2)+'\n');print('Built differential and planning kernels.')
if __name__=='__main__':main()
