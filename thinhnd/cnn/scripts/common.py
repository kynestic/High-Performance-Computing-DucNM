"""Phần dùng chung của các script thực nghiệm: gán lõi, chạy chương trình, đọc kết quả."""
import ctypes
import json
import re
import statistics
import subprocess
from ctypes import wintypes
from pathlib import Path

CNN = Path(__file__).resolve().parent.parent
RESULTS = CNN.parent / 'results'
MPIEXEC = r'C:\Program Files\Microsoft MPI\Bin\mpiexec.exe'
P_CORES = [4, 6, 8, 10, 2, 0]  # CPU logic đầu của mỗi lõi P trên i7-13620H
EPOCH = re.compile(r'epoch=(\d+) loss=(\S+) val=(\S+) test=(\S+) time=(\S+)')

_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]


def pin(cores):
    """Giới hạn vào `cores` lõi P (cả hai luồng phần cứng); tiến trình con thừa kế."""
    mask = sum(3 << cpu for cpu in P_CORES[:cores])
    if not _kernel32.SetProcessAffinityMask(_kernel32.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())


def train(program, ranks=1, **options):
    """Chạy một lần huấn luyện, trả về danh sách số đo theo epoch."""
    exe = str(CNN / 'bin' / f'train_{program}.exe')
    cmd = [MPIEXEC, '-n', str(ranks), exe] if program == 'mpi' else [exe]
    for key, value in options.items():
        cmd += [f'--{key}', str(value)]
    pin(ranks)
    out = subprocess.run(cmd, cwd=CNN, capture_output=True, text=True, timeout=600)
    if out.returncode:
        raise RuntimeError(out.stdout + out.stderr)
    return [dict(epoch=int(e), loss=float(loss), val=float(val), test=float(test), time=float(t))
            for e, loss, val, test, t in EPOCH.findall(out.stdout)]


def first_reach(epochs, target):
    """(epoch, thời gian huấn luyện cộng dồn) khi validation đạt ngưỡng lần đầu, hoặc None."""
    elapsed = 0.0
    for row in epochs:
        elapsed += row['time']
        if row['val'] >= target:
            return row['epoch'], elapsed
    return None


def mean_std(values):
    values = list(values)
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def number(x, digits=2):
    return f'{x:.{digits}f}'.replace('.', ',')


def save(name, data, lines):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f'{name}.json').write_text(json.dumps(data, indent=1), encoding='utf-8', newline='\n')
    (RESULTS / f'{name}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
