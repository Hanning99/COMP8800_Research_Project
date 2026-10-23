"""Create and verify an explicit, amended pre-execution lock."""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from scd_common import PARENT, sha256, write_pretty_json

def verify_lock(root, event=None):
    path=root/'preregistration/SC_D_PREREGISTRATION_LOCK.json'
    if not path.exists(): raise RuntimeError('CONFIRMATORY_SC_D_LOCKED_UNTIL_D4')
    lock=json.loads(path.read_text())
    if lock['status']!='FROZEN_WITH_DISCLOSED_PRELOCK_DEVIATION': raise RuntimeError('D4_NOT_FROZEN')
    for filename,digest in lock['files'].items():
        if sha256(Path(filename))!=digest: raise RuntimeError('LOCK_HASH_MISMATCH:'+filename)
    if event is not None:
        if event not in lock['events']: raise RuntimeError('EVENT_NOT_REGISTERED')
        for filename,digest in lock['events'][event]['files'].items():
            if sha256(Path(filename))!=digest: raise RuntimeError('EVENT_HASH_MISMATCH:'+filename)
    return lock

def main():
    root=Path(__file__).resolve().parent.parent
    lock_path=root/'preregistration/SC_D_PREREGISTRATION_LOCK.json'
    if lock_path.exists(): raise RuntimeError('REFUSING_TO_OVERWRITE_LOCK')
    gate=json.loads((root/'audits/AMENDED_PREEXECUTION_AUDIT.json').read_text())
    assert gate['pass'] and gate['prelock_deviation_disclosed']
    training_files=('run_sc_d.py','training_core.py','pdr_awac_adapter.py','reverse_pdr.py','scd_common.py','scoring_sc_d.py')
    assert all(gate['code_hashes'][name]==sha256(root/'src'/name) for name in training_files)
    gpu=json.loads((root/'audits/GPU_PREFLIGHT.json').read_text())
    assert gpu['pass'] and all(gpu['code_hashes'][name]==sha256(root/'src'/name) for name in training_files)
    from analyze_sc_d import auc, summary
    from verdict_sc_d import verdict
    assert auc({i:{'J_exact':i/1024} for i in range(0,1025,16)},'J_exact')==.5
    assert summary([1.]*50)['ci95']==[1.,1.]
    toy={k:summary([1.]*50) for k in ('J_exact','log_odds','gradient_cosine','gate_sequence_success','safe_route_success','full_task_success')}
    toy['weight_error']=summary([-1.]*50)
    assert verdict(toy,{'joint':50})=='SC_D_PDR_CLOSED_LOOP_REALIZATION_CONFIRMED'
    assert verdict(toy,{'joint':44})=='INCONCLUSIVE_DUE_TO_SC_D_MISSINGNESS'
    toy['safe_route_success']=summary([-.02]*50)
    assert verdict(toy,{'joint':50})=='SC_D_PDR_EARLY_GAIN_WITH_RELIABILITY_HARM'
    events={}
    for i in range(50):
        event=f'confirmatory_{i:03d}'
        source=PARENT/'source_events'/event
        reference=root/'references'/event
        m=json.loads((reference/'REFERENCE_MANIFEST.json').read_text())
        assert Path(m['reference_path']).resolve()==source/'old/checkpoint'
        assert sha256(reference/'pi_ref.npy')==m['pi_ref_file_sha256']
        assert sha256(reference/'qbar.npy')==m['qbar_file_sha256']
        paths=list(source.glob('old/checkpoint/*'))+list(source.glob('capability_checkpoint/*'))
        paths += [source/'STATUS.json',source/'source_manifest.json',source/'offline_dataset.jsonl',source/'acquisition_trace/online_stream.jsonl']
        paths += list(reference.glob('*'))+list((PARENT/'manifests/sc_c_panels'/event).glob('*.json'))
        paths += [PARENT/'sc_c_oracle'/event/'STALE/RUN_MANIFEST.json']
        events[event]={'seed':json.loads((source/'STATUS.json').read_text())['seed'],'replicate_id':'replicate_00','files':{str(p):sha256(p) for p in paths if p.is_file()}}
    files=list((root/'src').glob('*.py'))+[root/'manifests/support_graph.json',root/'preregistration/SC_D_PREREGISTRATION.md',root/'audits/AMENDED_PREEXECUTION_AUDIT.json',root/'audits/SC_D_PRELOCK_DEVIATION_001.md',root.parent/'PLAN.md']
    files+=list((PARENT/'src').rglob('*.py'))
    payload={'status':'FROZEN_WITH_DISCLOSED_PRELOCK_DEVIATION','generated_at_utc':datetime.now(timezone.utc).isoformat(),'events':events,'files':{str(p):sha256(p) for p in files},'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda},'total_cycles':2048,'intervention_end':1024,'evaluation_grid':list(range(0,2049,16)),'additional_pdr_checkpoints':[1,1025],'scoring':'post-training immutable snapshots only','comparator':'archived SC-C STALE; independently rescore its snapshots','bootstrap_seed':20260907,'bootstrap_resamples':20000,'prelock_deviation_disclosed':True,'complete_unexposed_preregistration_claim':False}
    write_pretty_json(lock_path,payload)
    print('D4 frozen with disclosed deviation; 50 events',flush=True)

if __name__=='__main__': main()
