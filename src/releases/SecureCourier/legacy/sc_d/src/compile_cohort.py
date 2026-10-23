"""Compile frozen references in ledger order; never perform actor updates."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import subprocess
import sys
from pathlib import Path
from scd_common import PARENT, sha256, read_json, write_pretty_json

ROOT = Path(__file__).resolve().parent.parent
records = []
for index in range(50):
    event = f'confirmatory_{index:03d}'
    source = PARENT / 'source_events' / event
    output = ROOT / 'references' / event
    if not (output / 'REFERENCE_MANIFEST.json').exists():
        subprocess.run([sys.executable, str(ROOT / 'src/compile_pdr_reference.py'), '--graph', str(ROOT / 'manifests/support_graph.json'), '--reference', str(source / 'old/checkpoint'), '--output', str(output), '--device', 'cpu'], check=True)
    manifest = read_json(output / 'REFERENCE_MANIFEST.json')
    assert manifest['reference_path'] == str(source / 'old/checkpoint')
    for name, digest in manifest['reference_files'].items():
        assert sha256(source / 'old/checkpoint' / name) == digest
    assert sha256(output / 'pi_ref.npy') == manifest['pi_ref_file_sha256']
    assert sha256(output / 'qbar.npy') == manifest['qbar_file_sha256']
    records.append({'event_id': event, 'reference_manifest': str(output / 'REFERENCE_MANIFEST.json'), 'sha256': sha256(output / 'REFERENCE_MANIFEST.json')})
    write_pretty_json(ROOT / 'manifests/reference_compilation_progress.json', {'completed': len(records), 'total': 50})
    print(f'{event}: reference verified ({len(records)}/50)', flush=True)
write_pretty_json(ROOT / 'manifests/reference_artifacts.json', records)
