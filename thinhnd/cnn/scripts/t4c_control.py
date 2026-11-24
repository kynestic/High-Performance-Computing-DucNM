"""T4 đối chứng: cùng số epoch và seed, chỉ đổi bước hoặc momentum (khoảng 2 phút)."""
from common import mean_std, number, save, train

DELAY = 15
STEPS = (0.02, 0.01, 0.005)  # bước hiệu dụng lr / (1 - momentum)
MOMENTA = (0.0, 0.9)
SEEDS = tuple(range(11, 21))
EPOCHS = 15
LEARNED = 70.0


def main():
    cell = {}
    for step in STEPS:
        for momentum in MOMENTA:
            lr = round(step * (1 - momentum), 8)
            cell[step, momentum] = [train('cuda', epochs=EPOCHS, lr=lr, momentum=momentum, delay=DELAY, seed=s)
                                    for s in SEEDS]
            print(f'step={step} momentum={momentum} done', flush=True)

    lines = ['# T4 đối chứng: tách ảnh hưởng của bước, số epoch và momentum', '',
             f'd = {DELAY}, {len(SEEDS)} seed, {EPOCHS} epoch. Mọi dòng dùng cùng seed. Accuracy validation.', '',
             '| Bước hiệu dụng | Momentum | Sau 5 epoch | Sau 15 epoch | Seed trên 70% ở epoch 15 |',
             '|--:|--:|--:|--:|--:|']
    rows = []
    for (step, momentum), runs in cell.items():
        early = mean_std(r[4]['val'] for r in runs)
        last = mean_std(r[-1]['val'] for r in runs)
        learned = sum(r[-1]['val'] > LEARNED for r in runs)
        rows.append(dict(delay=DELAY, step=step, momentum=momentum, seeds=SEEDS, runs=runs,
                         val5=early, val15=last, learned=learned))
        lines.append(f'| {number(step, 3)} | {number(momentum, 1)} | {number(early[0], 1)} ± {number(early[1], 1)} '
                     f'| {number(last[0], 1)} ± {number(last[1], 1)} | {learned}/{len(SEEDS)} |')

    lines += ['', '## Momentum 0 so với 0,9, ghép cặp theo seed (epoch 15)', '',
              '| Bước hiệu dụng | Hiệu trung bình (điểm) | Momentum 0 tốt hơn |', '|--:|--:|--:|']
    for step in STEPS:
        diff = [a[-1]['val'] - b[-1]['val'] for a, b in zip(cell[step, 0.0], cell[step, 0.9])]
        mean, std = mean_std(diff)
        lines.append(f'| {number(step, 3)} | {number(mean, 1)} ± {number(std, 1)} '
                     f'| {sum(x > 0 for x in diff)}/{len(SEEDS)} seed |')
    save('t4c_control', rows, lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
