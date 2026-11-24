# T8: đồng bộ và bất đồng bộ, so công bằng

6 rank (bất đồng bộ: 1 server và 5 worker), 5 seed, tối đa 20 epoch (tuần tự 12).
Ngưỡng: validation 85%. Learning rate chọn trên validation bằng seed 101, 102, lưới tự nới.
Test lấy ở epoch có validation cao nhất.

| Phương pháp | lr | Seed đạt | Epoch cần (trung vị) | Thời gian tới ngưỡng (s) | Tăng tốc tới ngưỡng | Tăng tốc mỗi epoch | Test |
|---|--:|--:|--:|--:|--:|--:|--:|
| C tuần tự, momentum 0,9 | 0.01 | 5/5 | 3 | 7,74 ± 1,19 | 1,00 | 1,00 | 85,77 ± 0,55% |
| CUDA, momentum 0,9 | 0.01 | 5/5 | 3 | 0,24 ± 0,04 | 32,90 | 35,80 | 85,95 ± 0,45% |
| Đồng bộ, momentum 0,9 | 0.01 | 5/5 | 5 | 2,73 ± 0,60 | 2,84 | 3,89 | 86,51 ± 0,35% |
| Đồng bộ, không momentum | 0.05 | 5/5 | 9 | 5,15 ± 0,72 | 1,50 | 3,88 | 86,28 ± 0,78% |
| Bất đồng bộ, momentum 0,9 | 0.01 | 4/5 | 14,5 | – | – | 4,54 | 84,40 ± 1,10% |
| Bất đồng bộ, không momentum | 0.05 | 5/5 | 14 | 7,38 ± 1,11 | 1,05 | 4,50 | 84,98 ± 0,59% |
| Trung bình tham số, momentum 0,9 | 0.02 | 5/5 | 8 | 4,09 ± 0,79 | 1,89 | 5,04 | 86,34 ± 0,43% |

## Ghép cặp theo seed: thời gian tới ngưỡng

| So sánh A với B | Cả hai cùng đạt | A nhanh hơn | Tỉ lệ thời gian B/A |
|---|--:|--:|--:|
| đồng bộ với bất đồng bộ, momentum 0,9 | 4/5 | 4/4 | 2,87 (1,92–4,61) |
| đồng bộ với bất đồng bộ, không momentum | 5/5 | 5/5 | 1,46 (1,11–1,77) |
| đồng bộ: có với không momentum | 5/5 | 5/5 | 2,00 (1,26–3,00) |
| bất đồng bộ: có với không momentum | 4/5 | 3/4 | 1,02 (0,73–1,16) |
| đồng bộ với trung bình tham số, momentum 0,9 | 5/5 | 5/5 | 1,54 (1,19–2,00) |

## Dò learning rate

| Phương pháp | lr: epoch cần (không đạt = 21) / validation tốt nhất | Chọn | Còn ở mép lưới |
|---|---|--:|:-:|
| Đồng bộ, momentum 0,9 | 0.005: 4,5 / 88,2%; 0.01: 3,5 / 88,0%; 0.02: 4,0 / 88,1% | 0.01 | không |
| Đồng bộ, không momentum | 0.025: 8,0 / 86,5%; 0.05: 6,0 / 87,4%; 0.1: 7,5 / 87,6%; 0.2: 8,0 / 87,2% | 0.05 | không |
| Bất đồng bộ, momentum 0,9 | 0.0025: 19,5 / 84,7%; 0.005: 18,5 / 85,9%; 0.01: 16,0 / 86,0%; 0.02: 21,0 / 83,2% | 0.01 | không |
| Bất đồng bộ, không momentum | 0.025: 17,5 / 85,7%; 0.05: 15,0 / 86,0%; 0.1: 18,5 / 85,5% | 0.05 | không |
