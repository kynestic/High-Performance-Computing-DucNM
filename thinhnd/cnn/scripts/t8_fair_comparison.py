"""T8: đồng bộ so với bất đồng bộ, cả hai có và không momentum, lưới learning rate tự nới (8-12 phút).

    python scripts/t8_fair_comparison.py tune   # chọn learning rate trên validation
    python scripts/t8_fair_comparison.py run    # đánh giá bằng seed 11-15
"""
import json
import statistics
import sys

from common import RESULTS, first_reach, mean_std, number, save, train

TARGET = 85.0
EPOCHS = 20
SEQ_EPOCHS = 12
RANKS = 6
TUNE_SEEDS = (101, 102)
SEEDS = (11, 12, 13, 14, 15)
MAX_EXTRA = 5  # số lần nới lưới tối đa

# (khóa, tên, chế độ, momentum, lưới ban đầu)
VARIANTS = (
    ('sync_m9', 'Đồng bộ, momentum 0,9', 'sync', 0.9, (0.005, 0.01, 0.02)),
    ('sync_m0', 'Đồng bộ, không momentum', 'sync', 0.0, (0.05, 0.1, 0.2)),
    ('async_m9', 'Bất đồng bộ, momentum 0,9', 'async', 0.9, (0.0025, 0.005, 0.01)),
    ('async_m0', 'Bất đồng bộ, không momentum', 'async', 0.0, (0.025, 0.05, 0.1)),
)
AVG_LR = 0.02  # đã chọn ở T1, nằm giữa lưới


def score(mode, momentum, lr):
    runs = []
    for seed in TUNE_SEEDS:
        options = dict(epochs=EPOCHS, lr=lr, momentum=momentum, seed=seed)
        # sync tương đương toán học với SGD tuần tự nên dò trên GPU
        runs.append(train('cuda', **options) if mode == 'sync' else train('mpi', RANKS, mode=mode, **options))
    reach = [first_reach(r, TARGET) for r in runs]
    return dict(lr=lr, runs=runs,
                epochs_needed=statistics.fmean(r[0] if r else EPOCHS + 1 for r in reach),
                best_val=statistics.fmean(max(e['val'] for e in r) for r in runs))


def tune():
    out = {}
    for key, name, mode, momentum, grid in VARIANTS:
        table = [score(mode, momentum, lr) for lr in grid]
        extra = 0
        while True:
            table.sort(key=lambda t: t['lr'])
            best = min(range(len(table)), key=lambda i: (table[i]['epochs_needed'], -table[i]['best_val']))
            if 0 < best < len(table) - 1 or extra == MAX_EXTRA:
                break
            lr = table[0]['lr'] / 2 if best == 0 else table[-1]['lr'] * 2
            table.append(score(mode, momentum, lr))
            extra += 1
        out[key] = dict(name=name, mode=mode, momentum=momentum, table=table, lr=table[best]['lr'],
                        at_edge=best in (0, len(table) - 1))
        print(f"{key}: lr={out[key]['lr']} at_edge={out[key]['at_edge']} "
              + ' '.join(f"{t['lr']:g}:{t['epochs_needed']:g}/{t['best_val']:.1f}" for t in table), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 't8_tuning.json').write_text(json.dumps(out, indent=1), encoding='utf-8', newline='\n')


def evaluate():
    tuning = json.loads((RESULTS / 't8_tuning.json').read_text(encoding='utf-8'))
    base = dict(lr=tuning['sync_m9']['lr'], momentum=0.9)
    methods = [('seq', 'C tuần tự, momentum 0,9', 'seq', 1, SEQ_EPOCHS, base),
               ('cuda', 'CUDA, momentum 0,9', 'cuda', 1, EPOCHS, base)]
    for key, v in tuning.items():
        methods.append((key, v['name'], 'mpi', RANKS, EPOCHS,
                        dict(mode=v['mode'], lr=v['lr'], momentum=v['momentum'])))
    methods.append(('avg_m9', 'Trung bình tham số, momentum 0,9', 'mpi', RANKS, EPOCHS,
                    dict(mode='avg', lr=AVG_LR, momentum=0.9)))

    runs = {m[0]: [] for m in methods}
    for n, seed in enumerate(SEEDS):
        for key, _, program, ranks, epochs, options in methods if n % 2 == 0 else reversed(methods):
            runs[key].append(train(program, ranks, epochs=epochs, seed=seed, **options))
        print(f'seed {seed} done', flush=True)

    rows = {}
    for key, name, _, _, epochs, options in methods:
        reach = [first_reach(r, TARGET) for r in runs[key]]
        rows[key] = dict(name=name, epochs=epochs, options=options, runs=runs[key], reach=reach,
                         reached=sum(1 for r in reach if r),
                         epoch_time=statistics.median(e['time'] for r in runs[key] for e in r),
                         test=mean_std(max(r, key=lambda e: e['val'])['test'] for r in runs[key]))

    n = len(SEEDS)
    lines = ['# T8: đồng bộ và bất đồng bộ, so công bằng', '',
             f'6 rank (bất đồng bộ: 1 server và 5 worker), {n} seed, tối đa {EPOCHS} epoch (tuần tự {SEQ_EPOCHS}).',
             'Ngưỡng: validation 85%. Learning rate chọn trên validation bằng seed 101, 102, lưới tự nới.',
             'Test lấy ở epoch có validation cao nhất.', '',
             '| Phương pháp | lr | Seed đạt | Epoch cần (trung vị) | Thời gian tới ngưỡng (s) | Tăng tốc tới ngưỡng '
             '| Tăng tốc mỗi epoch | Test |',
             '|---|--:|--:|--:|--:|--:|--:|--:|']
    seq = rows['seq']
    seq_time = mean_std(r[1] for r in seq['reach'] if r)[0] if seq['reached'] == n else None
    for r in rows.values():
        hit = [x for x in r['reach'] if x]
        needed = f'{statistics.median(x[0] for x in hit):g}'.replace('.', ',') if hit else '–'
        spent = speed = '–'
        if r['reached'] == n:  # chỉ so thời gian khi mọi seed đều đạt
            m, s = mean_std(x[1] for x in hit)
            spent = f'{number(m)} ± {number(s)}'
            if seq_time:
                r['speedup_to_target'] = seq_time / m
                speed = number(r['speedup_to_target'])
        r['speedup_per_epoch'] = seq['epoch_time'] / r['epoch_time']
        lines.append(f"| {r['name']} | {r['options']['lr']:g} | {r['reached']}/{n} | {needed} | {spent} | {speed} "
                     f"| {number(r['speedup_per_epoch'])} | {number(r['test'][0])} ± {number(r['test'][1])}% |")

    lines += ['', '## Ghép cặp theo seed: thời gian tới ngưỡng', '',
              '| So sánh A với B | Cả hai cùng đạt | A nhanh hơn | Tỉ lệ thời gian B/A |', '|---|--:|--:|--:|']
    for a, b, text in (('sync_m9', 'async_m9', 'đồng bộ với bất đồng bộ, momentum 0,9'),
                       ('sync_m0', 'async_m0', 'đồng bộ với bất đồng bộ, không momentum'),
                       ('sync_m9', 'sync_m0', 'đồng bộ: có với không momentum'),
                       ('async_m9', 'async_m0', 'bất đồng bộ: có với không momentum'),
                       ('sync_m9', 'avg_m9', 'đồng bộ với trung bình tham số, momentum 0,9')):
        pairs = [(x[1], y[1]) for x, y in zip(rows[a]['reach'], rows[b]['reach']) if x and y]
        ratio = [y / x for x, y in pairs]
        shown = f'{number(mean_std(ratio)[0])} ({number(min(ratio))}–{number(max(ratio))})' if ratio else '–'
        lines.append(f'| {text} | {len(pairs)}/{n} | {sum(y > x for x, y in pairs)}/{len(pairs)} | {shown} |')

    lines += ['', '## Dò learning rate', '',
              '| Phương pháp | lr: epoch cần (không đạt = 21) / validation tốt nhất | Chọn | Còn ở mép lưới |',
              '|---|---|--:|:-:|']
    for v in tuning.values():
        cells = '; '.join(f"{t['lr']:g}: {number(t['epochs_needed'], 1)} / {number(t['best_val'], 1)}%"
                          for t in v['table'])
        lines.append(f"| {v['name']} | {cells} | {v['lr']:g} | {'có' if v['at_edge'] else 'không'} |")

    save('t8_fair_comparison', dict(target=TARGET, seeds=SEEDS, methods=rows), lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('tune', 'run'):
        sys.exit(__doc__)
    tune() if sys.argv[1] == 'tune' else evaluate()
