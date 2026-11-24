"""T4: staleness x learning rate x momentum, mô phỏng tất định trên GPU (khoảng 2 phút)."""
from common import mean_std, number, save, train

DELAYS = (0, 3, 7, 11, 15)
MOMENTA = (0.0, 0.9)
STEPS = (0.02, 0.05, 0.1, 0.2)  # bước hiệu dụng lr / (1 - momentum)
SEEDS = (11, 12, 13)
EPOCHS = 5


def main():
    rows = []
    for momentum in MOMENTA:
        for step in STEPS:
            for delay in DELAYS:
                lr = round(step * (1 - momentum), 8)
                runs = [train('cuda', epochs=EPOCHS, lr=lr, momentum=momentum, delay=delay, seed=s) for s in SEEDS]
                rows.append(dict(momentum=momentum, step=step, lr=lr, delay=delay, runs=runs,
                                 val=mean_std(r[-1]['val'] for r in runs)))
            print(f'momentum={momentum} step={step} done', flush=True)

    cell = {(r['momentum'], r['step'], r['delay']): r for r in rows}
    lines = ['# T4: staleness, learning rate và momentum', '',
             f'Accuracy validation sau {EPOCHS} epoch, trung bình ± độ lệch chuẩn của {len(SEEDS)} seed (đơn vị %).',
             'Đoán ngẫu nhiên là khoảng 10%. d là số bước trễ, tương đương W = d + 1 worker chạy xoay vòng.']
    for momentum in MOMENTA:
        lines += ['', f'## Momentum {number(momentum, 1)}', '',
                  '| Bước hiệu dụng | lr | ' + ' | '.join(f'd = {d}' for d in DELAYS) + ' |',
                  '|--:|--:|' + '--:|' * len(DELAYS)]
        for step in STEPS:
            values = [cell[momentum, step, d]['val'] for d in DELAYS]
            lr = cell[momentum, step, 0]['lr']
            lines.append(f'| {number(step)} | {number(lr, 3)} | '
                         + ' | '.join(f'{number(m, 1)} ± {number(s, 1)}' for m, s in values) + ' |')
    save('t4_staleness', rows, lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
