"""Build everything into bin/: python build.py  (Windows, WinLibs GCC, MS-MPI, CUDA Toolkit)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FLAGS = ['-std=gnu11', '-O3', '-ffast-math', '-Wall', '-Wextra', '-static']


def find_gcc():
    found = shutil.which('gcc')
    if not found:
        packages = Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft/WinGet/Packages'
        found = next(iter(packages.glob('BrechtSanders.WinLibs*/mingw64/bin/gcc.exe')), None)
    if not found:
        sys.exit('gcc not found')
    return Path(found)


def find_cuda():
    candidates = [Path(os.environ['CUDA_PATH'])] if os.environ.get('CUDA_PATH') else []
    candidates += sorted(Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA').glob('v*'), reverse=True)
    return next((c for c in candidates if (c / 'include/nvrtc.h').exists()), None)


def main():
    gcc = find_gcc()
    env = dict(os.environ, PATH=str(gcc.parent) + os.pathsep + os.environ['PATH'])
    (ROOT / 'bin').mkdir(exist_ok=True)

    def build(name, sources, extra=()):
        subprocess.run([str(gcc), *FLAGS, *sources, *extra, '-o', f'bin/{name}.exe'], cwd=ROOT, env=env, check=True)
        print('OK', name, flush=True)

    model = 'src/common/model.c'
    build('train_seq', ['src/seq/train_seq.c', model])

    sdk = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Microsoft SDKs/MPI'
    build('train_mpi', ['src/mpi/train_mpi.c', model],
          ['-isystem', str(sdk / 'Include'), str(sdk / 'Lib/x64/msmpi.lib')])

    cuda = find_cuda()
    if cuda:
        build('train_cuda', ['src/cuda/train_cuda.c', model],
              ['-isystem', str(cuda / 'include'), str(cuda / 'lib/x64/cuda.lib'), str(cuda / 'lib/x64/nvrtc.lib')])
    else:
        print('SKIP train_cuda: CUDA Toolkit not found')

    build('gradient_check', ['tests/gradient_check.c', model], ['-DUSE_DOUBLE'])


if __name__ == '__main__':
    main()
