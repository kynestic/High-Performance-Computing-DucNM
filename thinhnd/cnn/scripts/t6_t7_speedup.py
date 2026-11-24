"""T6: CPU so với GPU theo lớp. T7: strong scaling MPI và CUDA (khoảng 2 phút)."""
import re
import statistics
import subprocess

from common import CNN, MPIEXEC, mean_std, number, pin, save

REPEAT = 3
EPOCHS = 3


def run(cores, cmd):
    pin(cores)
    done = subprocess.run(cmd, cwd=CNN, capture_output=True, text=True, timeout=300)
    if done.returncode:
        raise RuntimeError(done.stdout + done.stderr)
    return done.stdout


def exe(name):
    return str(CNN / 'bin' / f'{name}.exe')


def layers():
    samples = {}
    for _ in range(REPEAT):
        for layer, size, cpu, gpu in re.findall(r'bench layer=(\w+) size=(\d+) cpu=(\S+) gpu=(\S+)',
                                                run(1, [exe('train_cuda'), '--bench'])):
            samples.setdefault((layer, int(size)), []).append((float(cpu) * 1e3, float(gpu) * 1e3))
    rows = [dict(layer=layer, size=size, runs=v,
                 cpu_ms=statistics.median(c for c, _ in v), gpu_ms=statistics.median(g for _, g in v),
                 ratio=mean_std(c / g for c, g in v))
            for (layer, size), v in samples.items()]

    lines = ['# T6: CPU một luồng so với GPU, theo lớp', '',
             f'Cùng một laptop, {REPEAT} lần chạy. Thời gian là trung vị, tỉ lệ là trung bình ± độ lệch chuẩn.',
             'Chỉ forward, dữ liệu đã nằm sẵn trên thiết bị. CPU là C `-O3 -ffast-math`, không BLAS.']
    for layer, title, column in (('conv', 'Tích chập 8 bộ lọc 5x5 trên ảnh 28x28', 'Batch'),
                                 ('dense', 'Kết nối đầy đủ n x n, một vector', 'n')):
        lines += ['', f'## {title}', '', f'| {column} | CPU (ms) | GPU (ms) | CPU/GPU |', '|--:|--:|--:|--:|']
        for r in rows:
            if r['layer'] == layer:
                lines.append(f"| {r['size']} | {number(r['cpu_ms'], 4)} | {number(r['gpu_ms'], 4)} "
                             f"| {number(r['ratio'][0])} ± {number(r['ratio'][1])} |")
    save('t6_layers', rows, lines)
    return lines


def scaling():
    jobs = [('C tuần tự', 1, [exe('train_seq')])]
    jobs += [('MPI đồng bộ', p, [MPIEXEC, '-n', str(p), exe('train_mpi')]) for p in (1, 2, 4, 6)]
    jobs += [('CUDA', 1, [exe('train_cuda')])]
    runs = [[] for _ in jobs]
    for r in range(REPEAT):
        for i in range(len(jobs)) if r % 2 == 0 else reversed(range(len(jobs))):
            _, p, cmd = jobs[i]
            out = run(p, cmd + ['--epochs', str(EPOCHS), '--lr', '0.01', '--seed', '11'])
            t, comm = re.search(r'summary time=(\S+) comm=(\S+)', out).groups()
            runs[i].append((float(t), float(comm)))
        print(f'round {r + 1}/{REPEAT}', flush=True)

    rows = []
    for (name, p, _), v in zip(jobs, runs):
        rows.append(dict(method=name, p=p, runs=v, time=statistics.median(t for t, _ in v),
                         low=min(t for t, _ in v), high=max(t for t, _ in v),
                         comm_share=statistics.median(c / t for t, c in v)))
    lines = ['# T7: strong scaling trên cùng một máy', '',
             f'7.000 ảnh, batch 64 (chia cho các rank), {EPOCHS} epoch, {REPEAT} lần chạy, mỗi rank một lõi P.',
             'Hiệu suất của CUDA không tính vì số lõi GPU và CPU không so trực tiếp được.', '',
             '| Phương pháp | p | Thời gian trung vị (s) | Khoảng | Tăng tốc | Hiệu suất | Chờ/truyền |',
             '|---|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        r['speedup'] = rows[0]['time'] / r['time']
        mpi = r['method'].startswith('MPI')
        eff = number(100 * r['speedup'] / r['p'], 1) + '%' if mpi else '–'
        comm = number(100 * r['comm_share'], 1) + '%' if mpi else '–'
        lines.append(f"| {r['method']} | {r['p']} | {number(r['time'], 3)} "
                     f"| {number(r['low'], 3)}–{number(r['high'], 3)} | {number(r['speedup'])} | {eff} | {comm} |")
    save('t7_scaling', rows, lines)
    return lines


if __name__ == '__main__':
    for text in (layers(), scaling()):
        print('\n'.join(text).encode('ascii', 'replace').decode())
