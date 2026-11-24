# T1: thời gian tới cùng accuracy

7.000 train, 1.000 validation, 2.000 test. 5 seed (11–15), tối đa 12 epoch, batch 64.
Learning rate chọn trên validation bằng seed 101, 102. Ngưỡng: validation 85%.
Test lấy ở epoch có validation cao nhất. Thời gian không gồm đánh giá.

| Phương pháp | p | lr | Đạt ngưỡng | Epoch cần (trung vị) | Thời gian tới ngưỡng (s) | Tăng tốc tới ngưỡng | Tăng tốc mỗi epoch | Test |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| C tuần tự | 1 | 0,01 | 5/5 | 3 | 7,68 ± 1,22 | 1,00 | 1,00 | 85,77 ± 0,55% |
| CUDA | 1 | 0,01 | 5/5 | 3 | 0,23 ± 0,04 | 32,88 | 35,13 | 85,58 ± 0,58% |
| MPI đồng bộ | 6 | 0,01 | 5/5 | 5 | 2,93 ± 0,73 | 2,62 | 3,43 | 86,20 ± 0,24% |
| MPI trung bình tham số | 6 | 0,02 | 5/5 | 8 | 4,49 ± 0,89 | 1,71 | 4,49 | 85,36 ± 0,41% |
| MPI bất đồng bộ | 6 | 0,05 | 1/5 | 12 (seed đạt) | 6,76 (chỉ 1 seed) | – | 4,08 | 83,78 ± 1,00% |

## Dò learning rate

| Chế độ | lr | Epoch cần (trung bình, không đạt = 13) | Validation tốt nhất |
|---|--:|--:|--:|
| sync | 0,01 (chọn) | 3,5 | 87,30% |
| sync | 0,02 | 4,0 | 88,10% |
| sync | 0,05 | 5,0 | 86,55% |
| avg | 0,01 | 12,0 | 85,15% |
| avg | 0,02 (chọn) | 9,0 | 86,95% |
| avg | 0,05 | 11,0 | 85,45% |
| async | 0,05 (chọn) | 11,0 | 85,25% |
| async | 0,1 | 12,5 | 84,55% |
| async | 0,2 | 11,5 | 48,80% |
