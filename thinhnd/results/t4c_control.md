# T4 đối chứng: tách ảnh hưởng của bước, số epoch và momentum

d = 15, 10 seed, 15 epoch. Mọi dòng dùng cùng seed. Accuracy validation.

| Bước hiệu dụng | Momentum | Sau 5 epoch | Sau 15 epoch | Seed trên 70% ở epoch 15 |
|--:|--:|--:|--:|--:|
| 0,020 | 0,0 | 41,4 ± 19,4 | 69,7 ± 21,3 | 9/10 |
| 0,020 | 0,9 | 34,1 ± 14,1 | 62,0 ± 19,3 | 6/10 |
| 0,010 | 0,0 | 64,9 ± 7,2 | 77,9 ± 2,4 | 10/10 |
| 0,010 | 0,9 | 60,0 ± 4,3 | 76,0 ± 2,5 | 10/10 |
| 0,005 | 0,0 | 69,0 ± 6,4 | 77,0 ± 2,3 | 10/10 |
| 0,005 | 0,9 | 62,5 ± 7,8 | 76,6 ± 1,9 | 10/10 |

## Momentum 0 so với 0,9, ghép cặp theo seed (epoch 15)

| Bước hiệu dụng | Hiệu trung bình (điểm) | Momentum 0 tốt hơn |
|--:|--:|--:|
| 0,020 | 7,7 ± 25,5 | 7/10 seed |
| 0,010 | 1,8 ± 3,5 | 7/10 seed |
| 0,005 | 0,4 ± 2,6 | 6/10 seed |
