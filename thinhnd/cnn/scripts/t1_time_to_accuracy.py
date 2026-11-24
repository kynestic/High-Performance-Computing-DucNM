"""T1: thời gian tới cùng accuracy, chọn learning rate trên validation (khoảng 6-8 phút)."""
import json
import statistics
import sys

from common import RESULTS, first_reach, mean_std, number, save, train

TARGET = 85.0
EPOCHS = 12
TUNE_SEEDS = (101, 102)
SEEDS = (11, 12, 13, 14, 15)
GRID = {'sync': (0.01, 0.02, 0.05), 'avg': (0.01, 0.02, 0.05), 'async': (0.05, 0.1, 0.2)}
RANKS = 6


def tune(mode):
    table = []
    for lr in GRID[mode]:
        runs = []
        for seed in TUNE_SEEDS:
            # sync tương đương toán học với SGD tuần tự nên dò trên GPU cho nhanh
            if mode == 'sync':
                runs.append(train('cuda', epochs=EPOCHS, lr=lr, seed=seed))
            else:
                runs.append(train('mpi', RANKS, mode=mode, epochs=EPOCHS, lr=lr, seed=seed))
        reach = [first_reach(r, TARGET) for r in runs]
        table.append(dict(mode=mode, lr=lr, runs=runs,
                          epochs_needed=statistics.fmean(r[0] if r else EPOCHS + 1 for r in reach),
                          best_val=statistics.fmean(max(e['val'] for e in r) for r in runs)))
    return min(table, key=lambda t: (t['epochs_needed'], -t['best_val'])), table


def summarize(name, ranks, lr, runs):
    selected = [max(r, key=lambda e: e['val']) for r in runs]  # max() lấy epoch sớm nhất khi hòa
    reach = [first_reach(r, TARGET) for r in runs]
    hit = [r for r in reach if r]
    return dict(method=name, ranks=ranks, lr=lr, runs=runs, reached=len(hit),
                epochs_needed=statistics.median(r[0] for r in hit) if hit else None,
                time_to_target=mean_std(r[1] for r in hit) if hit else None,
                epoch_time=statistics.median(e['time'] for r in runs for e in r),
                best_epoch=statistics.median(e['epoch'] for e in selected),
                test=mean_std(e['test'] for e in selected))


def measure():
    chosen, tuning = {}, []
    for mode in GRID:
        best, table = tune(mode)
        chosen[mode] = best['lr']
        tuning += table
        print(f'tuned {mode}: lr={best["lr"]}', flush=True)

    methods = [
        ('C tuần tự', 'seq', 1, chosen['sync'], {}),
        ('CUDA', 'cuda', 1, chosen['sync'], {}),
        ('MPI đồng bộ', 'mpi', RANKS, chosen['sync'], {'mode': 'sync'}),
        ('MPI trung bình tham số', 'mpi', RANKS, chosen['avg'], {'mode': 'avg'}),
        ('MPI bất đồng bộ', 'mpi', RANKS, chosen['async'], {'mode': 'async'}),
    ]
    runs = [[] for _ in methods]
    for n, seed in enumerate(SEEDS):
        order = range(len(methods)) if n % 2 == 0 else reversed(range(len(methods)))
        for i in order:
            _, program, ranks, lr, extra = methods[i]
            runs[i].append(train(program, ranks, epochs=EPOCHS, lr=lr, seed=seed, **extra))
        print(f'seed {seed} done', flush=True)

    rows = [summarize(name, ranks, lr, r) for (name, _, ranks, lr, _), r in zip(methods, runs)]
    return chosen, tuning, rows


def report(chosen, tuning, rows):
    base = rows[0]
    lines = ['# T1: thời gian tới cùng accuracy', '',
             f'7.000 train, 1.000 validation, 2.000 test. {len(SEEDS)} seed (11–15), tối đa {EPOCHS} epoch, batch 64.',
             f'Learning rate chọn trên validation bằng seed 101, 102. Ngưỡng: validation {number(TARGET, 0)}%.',
             'Test lấy ở epoch có validation cao nhất. Thời gian không gồm đánh giá.', '',
             '| Phương pháp | p | lr | Đạt ngưỡng | Epoch cần (trung vị) | Thời gian tới ngưỡng (s) '
             '| Tăng tốc tới ngưỡng | Tăng tốc mỗi epoch | Test |',
             '|---|--:|--:|--:|--:|--:|--:|--:|--:|']
    for r in rows:
        spent = speed = needed = '–'
        r['speedup_to_target'] = None
        if r['reached']:
            m, sd = r['time_to_target']
            needed = f"{r['epochs_needed']:g}".replace('.', ',')
            if r['reached'] == len(SEEDS):  # chỉ so tốc độ khi mọi seed đều đạt ngưỡng
                spent = f'{number(m)} ± {number(sd)}'
                r['speedup_to_target'] = base['time_to_target'][0] / m
                speed = number(r['speedup_to_target'])
            else:
                spent = f"{number(m)} (chỉ {r['reached']} seed)"
                needed += ' (seed đạt)'
        r['speedup_per_epoch'] = base['epoch_time'] / r['epoch_time']
        lines.append(f"| {r['method']} | {r['ranks']} | {str(r['lr']).replace('.', ',')} | {r['reached']}/{len(SEEDS)} "
                     f"| {needed} | {spent} | {speed} | {number(r['speedup_per_epoch'])} "
                     f"| {number(r['test'][0])} ± {number(r['test'][1])}% |")

    lines += ['', '## Dò learning rate', '',
              '| Chế độ | lr | Epoch cần (trung bình, không đạt = 13) | Validation tốt nhất |', '|---|--:|--:|--:|']
    for t in tuning:
        mark = ' (chọn)' if chosen[t['mode']] == t['lr'] else ''
        lines.append(f"| {t['mode']} | {str(t['lr']).replace('.', ',')}{mark} | {number(t['epochs_needed'], 1)} "
                     f"| {number(t['best_val'])}% |")
    save('t1_time_to_accuracy', dict(target=TARGET, seeds=SEEDS, tuning=tuning, methods=rows), lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


def main():
    if '--report-only' in sys.argv:  # dựng lại bảng từ số đo đã lưu
        data = json.loads((RESULTS / 't1_time_to_accuracy.json').read_text(encoding='utf-8'))
        rows = data['methods']
        chosen = {'sync': rows[2]['lr'], 'avg': rows[3]['lr'], 'async': rows[4]['lr']}
        report(chosen, data['tuning'], rows)
    else:
        report(*measure())


if __name__ == '__main__':
    main()
