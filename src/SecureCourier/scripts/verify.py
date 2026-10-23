"""Verify every packaged source artifact and score-to-checkpoint association."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/integrity.json')
    a=p.parse_args()
    source=json.loads((ROOT/'provenance/source_files.json').read_text())
    manifest_path=ROOT/'provenance/release_files.json'
    files=json.loads(manifest_path.read_text()) if manifest_path.exists() else source
    def check(r):
        path=ROOT/r['path']
        assert path.is_file() and not path.is_symlink(),r['path']
        assert path.stat().st_size==r['size'] and sha256(path)==r['sha256'],r['path']
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(check,files))
    with (ROOT/'provenance/checkpoints.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==13000
    by_path={r['path']:r for r in source}
    for r in rows:
        state=r['checkpoint']+'/state.pt'
        assert by_path[state]['sha256']==r['state_sha256'],state
        score=json.loads((ROOT/r['score']).read_text())
        assert score['checkpoint_sha256']==r['state_sha256']
        assert score['cycle']==int(r['cycle']) and score['arm']==r['arm']
    for event in {r['event'] for r in rows}:
        run_manifests={}
        for arm in ['STANDARD_TD','FRESH_PDR']:
            observed={int(r['cycle']) for r in rows if r['event']==event and r['arm']==arm}
            required=set(range(0,2049,16))|({1,1025} if arm=='FRESH_PDR' else set())
            assert observed==required,(event,arm)
            run=ROOT/'records/runs'/event/arm
            rm=json.loads((run/'RUN_MANIFEST.json').read_text());run_manifests[arm]=rm
            assert rm['status']=='SEALED_COMPLETE'
            for name,key in [('continuation_stream.jsonl','continuation_stream_sha256'),('training_log.jsonl','training_log_sha256')]:
                relative=str((run/name).relative_to(ROOT))
                assert by_path[relative]['sha256']==rm[key],relative
            final=json.loads((ROOT/'checkpoints'/event/arm/'cycle_002048/manifest.json').read_text())
            for key in ['actor_hash','critic_hash','target_hash','actor_optimizer_hash','critic_optimizer_hash']:
                assert final[key]==rm['final'][key],(event,arm,key)
        for key in ['actor_hash','critic_hash','target_hash']:
            assert run_manifests['STANDARD_TD']['initial'][key]==run_manifests['FRESH_PDR']['initial'][key]
        source_dir=ROOT/'inputs/events'/event
        for relative in ['old/checkpoint','capability_checkpoint']:
            manifest=json.loads((source_dir/relative/'manifest.json').read_text())
            assert sha256(source_dir/relative/'state.pt')==manifest['state_sha256']
        ref=ROOT/'inputs/references'/event
        rm=json.loads((ref/'REFERENCE_MANIFEST.json').read_text())
        assert sha256(ROOT/'inputs/support_graph.json')==rm['graph_sha256']
        for name in ['qbar','pi_ref']:
            assert sha256(ref/(name+'.npy'))==rm[name+'_file_sha256']
    report=dict(status='PASS',files_verified=len(files),original_files=len(source),
        score_checkpoint_links=len(rows),events=50,regular_grid_points_per_arm=129,
        additional_pdr_cycles=[1,1025],no_external_symlinks=True,
        original_run_streams_and_final_models_match=True,paired_initial_models_match=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
