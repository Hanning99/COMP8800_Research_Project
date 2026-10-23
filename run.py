"""Evaluate saved actors, rebuild results or continue training for one benchmark.

    python run.py evaluate DeepSea
    python run.py results  DeepSea
    python run.py train    DeepSea -- --size 10 --method PDR

Checkpoints for the chosen benchmark are fetched from Hugging Face on first use.
"""
import argparse, hashlib, json, os, shutil, sqlite3, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = 'denghanning/COMP8800_Research_Project'
REVISION = 'cf78433d7a481e89260f6f86da89177f142d9524'
BENCHMARKS = ['DeepSea', 'KeyCorridor', 'MiniGridLoCA', 'SecureCourier']


# ---------------------------------------------------------------- data

def fetch(benchmark, data):
    """Download only the byte ranges of the archive that belong to `benchmark`."""
    import requests
    from huggingface_hub import hf_hub_download, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    index = Path(hf_hub_download(REPO, 'manifest.sqlite', repo_type='dataset', revision=REVISION,
                                 local_dir=data / 'index'))
    db = sqlite3.connect(f'file:{index}?mode=ro', uri=True)
    files = db.execute('SELECT path, sha256, bytes, mode, mtime_ns FROM files WHERE path LIKE ?',
                       (benchmark + '/%',)).fetchall()
    objects = db.execute(
        'SELECT DISTINCT o.sha256, o.shard, o.offset, o.length FROM files f JOIN objects o ON f.sha256 = o.sha256 '
        'WHERE f.path LIKE ?', (benchmark + '/%',)).fetchall()

    store = data / 'objects'
    todo = [o for o in objects if not (store / o[0][:2] / (o[0] + '.zst')).exists()]
    # Merge neighbouring objects in the same shard into one request.
    spans = []
    for sha, shard, offset, length in sorted(todo, key=lambda o: (o[1], o[2])):
        if spans and spans[-1][0] == shard and offset - spans[-1][2] < 1 << 20:
            spans[-1][2] = offset + length
            spans[-1][3].append((sha, offset, length))
        else:
            spans.append([shard, offset, offset + length, [(sha, offset, length)]])

    total = sum(o[3] for o in todo)
    done = 0
    headers = build_hf_headers()

    def get(span):
        shard, start, end, parts = span
        url = hf_hub_url(REPO, shard, repo_type='dataset', revision=REVISION)
        for attempt in range(5):
            try:
                r = requests.get(url, headers={**headers, 'Range': f'bytes={start}-{end - 1}'}, timeout=300)
                r.raise_for_status()
                blob = r.content
                if len(blob) != end - start:
                    raise IOError('short read')
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        for sha, offset, length in parts:
            dest = store / sha[:2] / (sha + '.zst')
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix('.part')
            tmp.write_bytes(blob[offset - start:offset - start + length])
            tmp.replace(dest)
        return sum(p[2] for p in parts)

    if todo:
        print(f'Downloading {benchmark}: {total / 1e9:.2f} GB')
        with ThreadPoolExecutor(8) as pool:
            for n in pool.map(get, spans):
                done += n
                print(f'\r  {done / total:6.1%}', end='', flush=True)
        print()
    return files


def restore(benchmark, data):
    """Unpack and checksum every file of `benchmark` into data/<benchmark>."""
    import zstandard as zstd

    out = data / benchmark
    if (out / '.complete').exists():
        return out
    files = fetch(benchmark, data)
    if out.exists():
        shutil.rmtree(out)
    print(f'Restoring {benchmark}: {len(files)} files')
    first = {}
    for i, (path, sha, size, mode, mtime) in enumerate(files):
        dest = out / path[len(benchmark) + 1:]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if sha in first:
            shutil.copyfile(first[sha], dest)
        else:
            h = hashlib.sha256()
            with open(data / 'objects' / sha[:2] / (sha + '.zst'), 'rb') as f, \
                    zstd.ZstdDecompressor().stream_reader(f) as src, open(dest, 'wb') as w:
                for chunk in iter(lambda: src.read(8 << 20), b''):
                    h.update(chunk)
                    w.write(chunk)
            if h.hexdigest() != sha or dest.stat().st_size != size:
                raise RuntimeError(f'Checksum mismatch: {path}')
            first[sha] = dest
        dest.chmod(mode)
        os.utime(dest, ns=(mtime, mtime))
        if (i + 1) % 2000 == 0:
            print(f'  {i + 1}/{len(files)}', flush=True)
    (out / '.complete').touch()
    shutil.rmtree(data / 'objects', ignore_errors=True)
    return out


def prepare(benchmark, data):
    """Restore data and put this repository's code next to it."""
    root = restore(benchmark, data)
    code = ROOT / 'src' / benchmark
    for f in code.rglob('*'):
        if f.is_file():
            dest = root / f.relative_to(code)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
    return root


def build(root):
    if (root / 'scripts' / 'build.py').exists() and not (root / 'build' / 'planning.so').exists():
        subprocess.run([sys.executable, str(root / 'scripts' / 'build.py')], cwd=root, check=True)


# ---------------------------------------------------------------- evaluation

def evaluate(benchmark, root, out, limit=None, representative=False, tolerance=1e-5):
    """Rebuild each final actor and compare its exact metrics with the recorded curve."""
    import numpy as np, pandas as pd, torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(root / 'vendor'))

    if benchmark == 'DeepSea':
        import deepsea_experiment as core
        core.DEVICE = torch.device('cpu')
        graphs = {}

        def score(r, state):
            key = (r['N'], r['seed'])
            if key not in graphs:
                graphs[key] = dict(np.load(root / 'inputs/deepsea/worlds' / f'N{key[0]}' / f'seed_{key[1]}' / 'graph.npz'))
            g = graphs[key]
            actor = core.Actor(g['features'].shape[1])
            actor.load_state_dict(state['models'][0])
            return core.exact_metrics(core.actor_policy(actor, g), g)
    elif benchmark == 'KeyCorridor':
        import key_exact as core
        core.DEVICE = core.base.DEVICE = torch.device('cpu')
        graphs = {}

        def score(r, state):
            task = r['cell']['task']
            if task not in graphs:
                graphs[task] = core.load_graph(task)
            g = graphs[task]
            actor = core.base.Actor(g['features'].shape[1])
            actor.load_state_dict(state['models'][0])
            policy = core.actor_policy(actor, torch.as_tensor(g['features']), torch.as_tensor(g['mask']))
            return core.exact_metrics(policy, g, core.TASKS[task][1])
    else:
        os.environ['MINIGRID_LOCA_PROTOCOL'] = str(root / 'inputs/loca_exact/PROTOCOL_LOCK.json')
        from run_study import Actor, extract_policy, GAMMA
        from minigrid_loca_core import build_graph, evaluate_policy
        g = build_graph()
        states = torch.arange(g.n_states)

        def score(r, state):
            actor = Actor().eval()
            actor.load_state_dict(state['models'][0])
            return evaluate_policy(g, extract_policy(actor, states), GAMMA)

    runs = json.loads((root / 'RUNS.json').read_text())
    if representative:
        chosen, seen = [], set()
        for r in runs:
            cell = r.get('cell', {})
            key = (cell.get('task'), r.get('N'), r.get('backbone', cell.get('backbone')), r['method'])
            if key not in seen:
                chosen.append(r)
                seen.add(key)
    else:
        chosen = runs[:limit]

    rows = []
    for i, r in enumerate(chosen):
        z = torch.load(root / r['final'], map_location='cpu', weights_only=False)
        metrics = score(r, z.get('state', z))
        curve = pd.read_csv(root / r['curve'])
        final = curve.loc[curve['update'].idxmax()]
        checked = [k for k in metrics if k in final.index and (benchmark != 'DeepSea' or k == 'expected_return')]
        error = max(abs(float(metrics[k]) - float(final[k])) for k in checked)
        if error > tolerance:
            raise RuntimeError(f'Run {i} ({r["method"]}, seed {r["seed"]}): metric differs from record by {error:.2e}')
        rows.append({'method': r['method'], 'seed': r['seed'], 'update': int(final['update']), 'error': error,
                     **{k: float(metrics[k]) for k in checked}})
        print(f'\r  evaluated {i + 1}/{len(chosen)}', end='', flush=True)
    print()
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / 'evaluation.csv', index=False)
    print(f'All {len(rows)} actors match their recorded metrics (max error {max(r["error"] for r in rows):.1e}).')


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['evaluate', 'results', 'train'])
    p.add_argument('benchmark', choices=BENCHMARKS)
    p.add_argument('--data', type=Path, default=ROOT / 'data', help='where checkpoints are stored (default: ./data)')
    p.add_argument('--out', type=Path, help='output directory (default: ./outputs/<benchmark>-<action>-<time>)')
    p.add_argument('--limit', type=int, help='evaluate only the first N runs')
    p.add_argument('--quick', action='store_true', help='evaluate one seed per setting and method')
    a, extra = p.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]

    data = a.data.resolve()
    root = prepare(a.benchmark, data)
    out = (a.out or ROOT / 'outputs' / f'{a.benchmark}-{a.action}-{time.strftime("%Y%m%d-%H%M%S")}').resolve()
    if out.exists():
        p.error(f'{out} already exists')

    env = {**os.environ, 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '1')}
    sc = a.benchmark == 'SecureCourier'
    if a.action == 'evaluate' and not sc:
        evaluate(a.benchmark, root, out, a.limit, a.quick)
        print('Saved', out)
        return
    if a.action == 'evaluate':
        events = ['--events', 'all'] if not a.quick else ['--cycles', '1024']
        cmd = ['scripts/rescore.py', '--output', str(out / 'rescored.csv'), *events, *extra]
    elif a.action == 'results':
        cmd = ['scripts/export_csv.py', '--output', str(out)] if sc else ['scripts/aggregate.py', '--out', str(out)]
    else:
        build(root)
        cmd = ['scripts/train.py', '--output', str(out)] if sc else ['scripts/reproduce.py', '--out', str(out)]
        cmd += extra
    subprocess.run([sys.executable, *cmd], cwd=root, env=env, check=True)
    print('Saved', out)


if __name__ == '__main__':
    main()
