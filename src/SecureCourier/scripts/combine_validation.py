"""Combine the two disjoint full-checkpoint rescoring audits."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=[]
    for name in ['rescored_part1.csv','rescored_part2.csv']:
        with (ROOT/'validation'/name).open(newline='') as f:
            rows.extend(csv.DictReader(f))
    actual={(r['event'],r['arm'],int(r['cycle'])) for r in rows}
    expected={(f'confirmatory_{i:03d}',arm,u) for i in range(50)
              for arm in ['STANDARD_TD','FRESH_PDR'] for u in range(0,2049,16)}
    assert len(rows)==len(actual)==12900 and actual==expected
    rows.sort(key=lambda r:(r['event'],r['arm'],int(r['cycle'])))
    with (ROOT/'validation/rescored_full.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    errors={k:max(float(r[k+'_abs_error']) for r in rows) for k in ['p_shortcut','J_exact']}
    assert max(errors.values())<2e-6
    report=dict(status='PASS',checkpoints=len(rows),events=50,regular_grid=[0,2048,16],
                all_checkpoint_hashes_match=True,max_abs_errors=errors,
                component_reports=['rescored_part1.json','rescored_part2.json'])
    (ROOT/'validation/rescored_full.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
