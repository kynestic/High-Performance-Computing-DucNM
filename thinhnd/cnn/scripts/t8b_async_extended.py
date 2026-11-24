"""T8 chạy bù: bất đồng bộ momentum 0,9 với tối đa 40 epoch (khoảng 4 phút).

Cả 5 seed đều chạy lại đúng một lượt, không thử lại seed nào: chỉ chạy lại seed chưa đạt sẽ thiên lệch,
vì bản bất đồng bộ không tất định. Đồng bộ được chạy xen kẽ để so thời gian trong cùng phiên.
"""
import statistics

from common import first_reach, mean_std, number, save, train

TARGET = 85.0
SEEDS = (11, 12, 13, 14, 15)
VARIANTS = (('sync', 'Đồng bộ', 20), ('async', 'Bất đồng bộ', 40))
LR, MOMENTUM, RANKS = 0.01, 0.9, 6


def main():
    runs = {mode: [] for mode, _, _ in VARIANTS}
    for n, seed in enumerate(SEEDS):
        for mode, _, epochs in VARIANTS if n % 2 == 0 else reversed(VARIANTS):
            run = train('mpi', RANKS, mode=mode, epochs=epochs, lr=LR, momentum=MOMENTUM, seed=seed)
            runs[mode].append(run)
            print(f'seed {seed} {mode}: reach={first_reach(run, TARGET)} max={max(e["val"] for e in run)}', flush=True)

    lines = ['# T8 chạy bù: momentum 0,9, bất đồng bộ tối đa 40 epoch', '',
             '6 rank, lr 0,01, seed 11-15, ngưỡng validation 85%. Mỗi seed chạy đúng một lượt.',
             'Test lấy ở epoch có validation cao nhất trong 20 epoch đầu, để cùng giới hạn với T8.', '',
             '| Phương pháp | Seed đạt | Epoch cần (trung vị) | Epoch cần từng seed | Thời gian tới ngưỡng (s) | Test |',
             '|---|--:|--:|--:|--:|--:|']
    rows = []
    for mode, name, epochs in VARIANTS:
        reach = [first_reach(r, TARGET) for r in runs[mode]]
        hit = [r for r in reach if r]
        test = mean_std(max(r[:20], key=lambda e: e['val'])['test'] for r in runs[mode])
        rows.append(dict(mode=mode, epochs=epochs, runs=runs[mode], reach=reach, test=test))
        spent = '–'
        if len(hit) == len(SEEDS):
            spent = ' ± '.join(number(x) for x in mean_std(r[1] for r in hit))
        needed = f'{statistics.median(r[0] for r in hit):g}'.replace('.', ',') if hit else '–'
        each = ', '.join(str(r[0]) if r else f'>{epochs}' for r in reach)
        lines.append(f'| {name} | {len(hit)}/{len(SEEDS)} | {needed} | {each} | {spent} '
                     f'| {number(test[0])} ± {number(test[1])}% |')
    pairs = [(a[1], b[1]) for a, b in zip(rows[0]['reach'], rows[1]['reach']) if a and b]
    ratio = [b / a for a, b in pairs]
    if ratio:
        lines += ['', f'Ghép cặp: đồng bộ nhanh hơn ở {sum(x > 1 for x in ratio)}/{len(ratio)} seed, tỉ lệ '
                  f'{number(mean_std(ratio)[0])} ({number(min(ratio))}–{number(max(ratio))}).']
    save('t8b_async_extended', rows, lines)
    print('\n'.join(lines).encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()
