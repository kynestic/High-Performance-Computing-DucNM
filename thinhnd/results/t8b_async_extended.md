# T8 chạy bù: momentum 0,9, bất đồng bộ tối đa 40 epoch

6 rank, lr 0,01, seed 11-15, ngưỡng validation 85%. Mỗi seed chạy đúng một lượt.
Test lấy ở epoch có validation cao nhất trong 20 epoch đầu, để cùng giới hạn với T8.

| Phương pháp | Seed đạt | Epoch cần (trung vị) | Epoch cần từng seed | Thời gian tới ngưỡng (s) | Test |
|---|--:|--:|--:|--:|--:|
| Đồng bộ | 5/5 | 5 | 3, 5, 6, 4, 5 | 2,77 ± 0,65 | 86,51 ± 0,35% |
| Bất đồng bộ | 5/5 | 16 | 14, 16, 16, 13, 28 | 9,08 ± 3,18 | 84,21 ± 1,63% |

Ghép cặp: đồng bộ nhanh hơn ở 5/5 seed, tỉ lệ 3,31 (2,33–4,70).
