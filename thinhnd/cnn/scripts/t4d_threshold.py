"""T4 đối chứng vùng ngưỡng: các ô tích từ 0,30 của lưới, chạy đủ 15 epoch (khoảng 2 phút)."""
from common import mean_std, number, save, train

CASES = ((3, 0.1), (7, 0.05), (11, 0.05), (3, 0.2), (7, 0.1), (15, 0.05))
SEEDS = tuple(range(11, 21))
EPOCHS = 15


def main():
    rows = []
    lines = ['# T4 đối chứng vùng ngưỡng', '',
             f'Momentum 0, {len(SEEDS)} seed, {EPOCHS} epoch. Accuracy validation.', '',
             '| d | Bước | Tích | Sau 5 epoch | Sau 15 epoch | Seed trên 70% ở epoch 15 |',
             '|--:|--:|--:|--:|--:|--:|']
    for delay, step in CASES:
        runs = [train('cuda', epochs=EPOCHS, lr=step, momentum=0, delay=delay, seed=s) for s in SEEDS]
        early = mean_std(r[4]['val'] for r in runs)
        last = mean_std(r[-1]['val'] for r in runs)
        learned = sum(r[-1]['val'] > 70 for r in runs)
        rows.append(dict(delay=delay, step=step, runs=runs, val5=early, val15=last, learned=learned))
        lines.append(f'| {delay} | {number(step)} | {number(delay * step)} | {number(early[0], 1)} ± {number(early[1], 1)} '
                     f'| {number(last[0], 1)} ± {number(last[1], 1)} | {learned}/{len(SEEDS)} |')
    save('t4d_threshold', rows, lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
