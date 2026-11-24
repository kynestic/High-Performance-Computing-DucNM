"""T4 bổ sung: kiểm chứng ngưỡng theo tích d x bước hiệu dụng (khoảng 20 giây)."""
from common import mean_std, number, save, train

CASES = ((15, 0.01), (15, 0.005), (31, 0.005), (31, 0.02))
SEEDS = (11, 12, 13)
EPOCHS = 15


def main():
    rows = []
    lines = ['# T4 bổ sung: ngưỡng theo tích d x bước', '',
             f'Momentum 0, {EPOCHS} epoch, {len(SEEDS)} seed, accuracy validation ở epoch cuối.', '',
             '| d | Bước | Tích | Validation |', '|--:|--:|--:|--:|']
    for delay, step in CASES:
        runs = [train('cuda', epochs=EPOCHS, lr=step, momentum=0, delay=delay, seed=s) for s in SEEDS]
        mean, std = mean_std(r[-1]['val'] for r in runs)
        rows.append(dict(delay=delay, step=step, runs=runs, val=(mean, std)))
        lines.append(f'| {delay} | {number(step, 3)} | {number(delay * step, 3)} | {number(mean, 1)} ± {number(std, 1)}% |')
    save('t4b_staleness_threshold', rows, lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
