"""Environment stamp for reproducibility: interpreter, platform, key package versions, BLAS/LAPACK backend, torch
threads/backend, thread env vars, git commit. Adapted from TheBridge tools/envstamp.py (commit 243ec82).
Usage: python tools/envstamp.py [out.json]   (or import envstamp; envstamp.stamp() -> dict to embed in results JSON)"""
import sys, json, platform, subprocess, importlib, os

PKGS = ['numpy', 'scipy', 'torch', 'pycbc', 'gwpy', 'lal', 'sbi', 'qnm', 'mpmath', 'ringdown', 'jax']
THREAD_VARS = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'PYTORCH_ENABLE_MPS_FALLBACK']


def _blas():
    out = {}
    for mod in ('numpy', 'scipy'):
        try:
            m = importlib.import_module(mod); cfg = m.show_config(mode='dicts')
            dep = cfg.get('Build Dependencies', {})
            out[mod] = {k: dep.get(k, {}).get('name') for k in ('blas', 'lapack')}
        except Exception as e:
            out[mod] = {'error': repr(e)}
    return out


def _torch():
    try:
        import torch
        return dict(threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                    mps=bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()),
                    cuda=torch.cuda.is_available(), default_dtype=str(torch.get_default_dtype()))
    except Exception as e:
        return {'error': repr(e)}


def _git(path='.'):
    try:
        c = subprocess.run(['git', '-C', path, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(['git', '-C', path, 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
        return {'commit': c, 'dirty': bool(d)}
    except Exception as e:
        return {'error': repr(e)}


def stamp(repo='.'):
    v = {}
    for p in PKGS:
        try: v[p] = getattr(importlib.import_module(p), '__version__', 'present')
        except Exception: v[p] = None
    return dict(python=sys.version.split()[0], executable=sys.executable, implementation=platform.python_implementation(),
                platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(), packages=v,
                blas_lapack=_blas(), torch=_torch(), thread_env={k: os.environ.get(k) for k in THREAD_VARS},
                git=_git(repo))


if __name__ == '__main__':
    s = stamp(); out = json.dumps(s, indent=1); print(out)
    if len(sys.argv) > 1: open(sys.argv[1], 'w').write(out + '\n')
