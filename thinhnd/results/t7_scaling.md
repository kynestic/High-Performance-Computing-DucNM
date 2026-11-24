# T7: strong scaling trên cùng một máy

7.000 ảnh, batch 64 (chia cho các rank), 3 epoch, 3 lần chạy, mỗi rank một lõi P.
Hiệu suất của CUDA không tính vì số lõi GPU và CPU không so trực tiếp được.

| Phương pháp | p | Thời gian trung vị (s) | Khoảng | Tăng tốc | Hiệu suất | Chờ/truyền |
|---|--:|--:|--:|--:|--:|--:|
| C tuần tự | 1 | 6,900 | 6,735–6,929 | 1,00 | – | – |
| MPI đồng bộ | 1 | 6,702 | 6,550–6,800 | 1,03 | 102,9% | 0,0% |
| MPI đồng bộ | 2 | 3,673 | 3,449–3,850 | 1,88 | 93,9% | 2,2% |
| MPI đồng bộ | 4 | 2,111 | 2,050–2,290 | 3,27 | 81,7% | 13,5% |
| MPI đồng bộ | 6 | 1,932 | 1,812–2,053 | 3,57 | 59,5% | 27,8% |
| CUDA | 1 | 0,209 | 0,208–0,212 | 33,05 | – | – |
