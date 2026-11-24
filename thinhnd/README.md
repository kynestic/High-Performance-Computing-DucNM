# CNN tự viết bằng C/MPI/CUDA: thời gian tới cùng accuracy và ảnh hưởng của gradient trễ

Phần việc của thư mục `thinhnd/`, bám bài V. Hegde, S. Usmani,
[*Parallel and Distributed Deep Learning*](https://hgpu.org/?p=16046) (Stanford, 2016).
Bài báo không nằm trong repo.

## Mục tiêu

1. Cùng một CNN, trên cùng một laptop: C tuần tự, MPI và CUDA nhanh hơn nhau bao nhiêu?
2. Đồng bộ, bất đồng bộ và trung bình tham số: cách nào tới **cùng accuracy** nhanh nhất khi được so công bằng?
3. Gradient đến trễ ảnh hưởng thế nào tới khả năng học, theo learning rate?

Không dùng BLAS, cuBLAS, cuDNN hay framework học sâu. Lan truyền ngược, SGD, cây broadcast/reduce và
kernel GPU đều được cài đặt trong thư mục này.

### Bài toán và thuật toán

Cho tập ảnh `(x_i, y_i)`, tìm tham số `w` cực tiểu hóa loss softmax bằng SGD theo mini-batch `B` có `m` ảnh:

```
L_i = -f_{y_i} + log(sum_j exp(f_j))

v <- mu * v + 1/m * sum_{i in B} grad L_i(w)
w <- w - lr * v
```

Mô hình có P = 54.314 tham số:

```
1x28x28 -> Conv 8@5x5  -> ReLU -> MaxPool
        -> Conv 16@5x5 -> ReLU -> MaxPool
        -> Dense 64 -> ReLU -> Dense 10 -> Softmax
```

| Chương trình | Thiết kế song song | Truyền thông mỗi epoch (n ảnh, batch b, p rank) |
|---|---|---|
| `train_seq` | C tuần tự, làm mốc | – |
| `train_mpi --mode sync` | Chia mini-batch cho các rank, cộng gradient theo cây nhị thức | n/b vòng, mỗi vòng O(P log p) |
| `train_mpi --mode avg` | Mỗi rank chạy SGD trên phần của mình, cuối epoch lấy trung bình | 1 vòng O(P log p) |
| `train_mpi --mode async` | Rank 0 là parameter server, worker không chờ nhau | n/b cặp thông điệp O(P) |
| `train_cuda` | Một luồng GPU tính một phần tử đầu ra; gradient conv theo map rồi reduce | chép hoán vị và loss |

## Cấu trúc thư mục

```
thinhnd/
├── README.md
├── PLAN.md          kế hoạch và dự đoán của từng thí nghiệm
├── cnn/
│   ├── build.py
│   ├── src/common/  mô hình, lan truyền ngược, đọc dữ liệu
│   ├── src/seq/     mã C tuần tự
│   ├── src/mpi/     mã song song MPI
│   ├── src/cuda/    mã song song CUDA (host C và kernels.cu)
│   ├── tests/       kiểm tra gradient bằng sai phân hữu hạn
│   └── scripts/     script thí nghiệm
└── results/         bảng .md và số đo gốc .json
```

## Cách chạy lại

Đã thử trên Windows 11, i7-13620H, RTX 4060 Laptop 8 GB, GCC 16.2, MS-MPI 10.1, CUDA Toolkit 12.8.
Chưa thử trên Linux.

**Cài đặt** (không cần Visual Studio vì kernel được biên dịch bằng NVRTC):

```powershell
winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e
winget install --id Microsoft.msmpi -e
winget install --id Microsoft.msmpisdk -e
```

Cài thêm CUDA Toolkit 12.x và Python 3 (chỉ dùng thư viện chuẩn, cho script build và thực nghiệm).

**Dữ liệu:** tải hai file của Fashion-MNIST (4,4 MB), giải nén vào `thinhnd/cnn/data/`:

- http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
- http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

**Biên dịch, kiểm tra và chạy:**

```powershell
cd thinhnd/cnn
python build.py

# kiểm tra
.\bin\gradient_check.exe
.\bin\train_cuda.exe --check

# tuần tự
.\bin\train_seq.exe --epochs 3

# song song
mpiexec -n 4 .\bin\train_mpi.exe --mode sync --epochs 3
mpiexec -n 4 .\bin\train_mpi.exe --mode avg --epochs 3
mpiexec -n 4 .\bin\train_mpi.exe --mode async --epochs 3
.\bin\train_cuda.exe --epochs 3

# thực nghiệm
python scripts\t1_time_to_accuracy.py        # khoảng 8 phút
python scripts\t4_staleness.py               # khoảng 2 phút
python scripts\t4b_staleness_threshold.py    # khoảng 20 giây
python scripts\t4c_control.py                # khoảng 2 phút
python scripts\t4d_threshold.py              # khoảng 2 phút
python scripts\t6_t7_speedup.py              # khoảng 2 phút
python scripts\t8_fair_comparison.py tune    # khoảng 5 phút
python scripts\t8_fair_comparison.py run     # khoảng 11 phút
python scripts\t8b_async_extended.py        # khoảng 5 phút
```

Tùy chọn: `--epochs`, `--batch`, `--train`, `--lr`, `--momentum`, `--seed`, `--data`, và `--delay` (chỉ CUDA).
Sơ đồ lõi trong `scripts/common.py` dành cho i7-13620H, máy khác cần sửa `P_CORES`.

## Kết quả

Đo ngày 22–23/11/2026 trên máy ở mục trên. Điều kiện chung: 10.000 ảnh Fashion-MNIST chia 7.000 train, 1.000 validation, 2.000 test; batch 64.

### 1. Thời gian tới cùng accuracy (T8)

5 seed, ngưỡng validation 85%, tối đa 20 epoch (tuần tự 12, dòng † là 40). MPI dùng 6 rank, bản bất đồng bộ gồm 1 server
và 5 worker. Đồng bộ và bất đồng bộ đều được chạy **có và không momentum**, learning rate của từng biến thể
được chọn trên validation và **không nằm ở mép lưới** ([chi tiết](results/t8_fair_comparison.md)).

| Phương pháp | Momentum | Seed đạt | Epoch cần | Thời gian tới ngưỡng (s) | Tăng tốc tới ngưỡng | Tăng tốc mỗi epoch | Test |
|---|--:|--:|--:|--:|--:|--:|--:|
| C tuần tự | 0,9 | 5/5 | 3 | 7,74 ± 1,19 | 1,00 | 1,00 | 85,77% |
| **CUDA** | 0,9 | 5/5 | 3 | **0,24 ± 0,04** | **32,90** | 35,80 | 85,95% |
| MPI đồng bộ | 0,9 | 5/5 | 5 | 2,73 ± 0,60 | 2,84 | 3,89 | 86,51% |
| MPI đồng bộ | 0 | 5/5 | 9 | 5,15 ± 0,72 | 1,50 | 3,88 | 86,28% |
| MPI bất đồng bộ † | 0,9 | 5/5 | 16 | 9,08 ± 3,18 | 0,85 | 4,42 | 84,21% |
| MPI bất đồng bộ | 0 | 5/5 | 14 | 7,38 ± 1,11 | 1,05 | 4,50 | 84,98% |
| MPI trung bình tham số | 0,9 | 5/5 | 8 | 4,09 ± 0,79 | 1,89 | 5,04 | 86,34% |

† Trong 20 epoch, biến thể này có một seed chỉ lên tới 84,1%. Vì bản bất đồng bộ không tất định, cả 5 seed được
chạy lại đúng một lượt với tối đa 40 epoch thay vì chỉ chạy lại seed đó; seed chậm nhất cần 28 epoch
([chi tiết](results/t8b_async_extended.md)). Test vẫn lấy trong 20 epoch đầu. Bản đồng bộ chạy xen kẽ trong lượt
này chỉ lệch 1,4% so với lượt trước, nên vẫn dùng mốc tuần tự của lượt trước.

- **So cùng điều kiện, đồng bộ nhanh hơn bất đồng bộ.** Không momentum: nhanh hơn 1,46 lần, ở 5/5 seed.
  Momentum 0,9: nhanh hơn 3,31 lần, cũng ở 5/5 seed (hai bên đo trong cùng một lượt).
- **Khoảng cách phụ thuộc momentum, nên phải so ở cùng mức momentum.** Momentum giúp bản đồng bộ nhanh
  gấp 2,0 lần (trung bình theo cặp seed) nhưng không giúp bản bất đồng bộ. Vì thế chênh lệch tăng từ 1,46 lên 3,31 lần.
- **Momentum không cho thấy lợi ích với bản bất đồng bộ:** có momentum chỉ tới ngưỡng sớm hơn ở 2/5 seed.
- **Bất đồng bộ 6 rank tới 85% xấp xỉ một lõi tuần tự:** tăng tốc 0,85 (có momentum) và 1,05
  (không momentum). Số 0,85 bị kéo xuống bởi seed cần 28 epoch; xét từng seed, bản có momentum vẫn nhanh hơn
  tuần tự ở 3/5 seed, nên không kết luận là nó chậm hơn tuần tự.
- **Nhanh mỗi epoch không có nghĩa là xong sớm.** Trung bình tham số và bất đồng bộ nhanh hơn đồng bộ ở mỗi
  epoch, nhưng cần nhiều epoch hơn nên tới 85% chậm hơn bản đồng bộ có momentum.

### 2. Gradient trễ và learning rate (T4)

Mô phỏng tất định: gradient tính ở bước t được áp ở bước t + d. Gọi η = lr/(1 − momentum) là bước hiệu dụng.
Bảng dưới: momentum 0, 15 epoch, 10 seed; một seed được tính là học được nếu validation cuối trên 70%
([đối chứng d = 15](results/t4c_control.md), [vùng ngưỡng](results/t4d_threshold.md)).

| d | η | d × η | Validation | Seed học được |
|--:|--:|--:|--:|--:|
| 15 | 0,005 | 0,075 | 77,0 ± 2,3% | 10/10 |
| 15 | 0,01 | 0,15 | 77,9 ± 2,4% | 10/10 |
| 15 | 0,02 | 0,30 | 69,7 ± 21,3% | 9/10 |
| 3 | 0,10 | 0,30 | 61,8 ± 35,8% | 7/10 |
| 7 | 0,05 | 0,35 | 62,4 ± 31,0% | 6/10 |
| 11 | 0,05 | 0,55 | 42,9 ± 35,1% | 4/10 |
| 3 | 0,20 | 0,60 | 15,3 ± 16,7% | 0/10 |
| 7 | 0,10 | 0,70 | 15,4 ± 18,4% | 0/10 |
| 15 | 0,05 | 0,75 | 30,4 ± 27,9% | 1/10 |

- Trong các cấu hình đã đo, tích d × η càng lớn thì càng dễ hỏng, nhưng **không có ngưỡng sắc**. Hai ô cùng
  tích 0,30 cho 9/10 và 7/10, nên tích này cũng không phải yếu tố duy nhất.
- Ở d = 15, giảm η từ 0,02 xuống 0,01 (cùng seed, cùng số epoch) làm kết quả ổn định hơn: 9/10 lên 10/10
  seed, độ lệch chuẩn 21,3 xuống 2,4 điểm.
- **Số epoch quan trọng.** Ô d = 15, η = 0,02 chỉ đạt 41,4% sau 5 epoch nhưng 69,7% sau 15 epoch.
  [Lưới 5 epoch](results/t4_staleness.md) vì vậy không dùng được để kết luận cấu hình nào phân kỳ.
- **Chưa kết luận được momentum chịu trễ kém hơn:** ở cùng η, chênh 0,4–7,7 điểm, nhỏ hơn độ lệch chuẩn
  của hiệu (2,6–25,5 điểm).

### 3. CPU so với GPU theo lớp (T6)

CPU một luồng, 3 lần chạy ([chi tiết](results/t6_layers.md)).

| Lớp | Kích thước nhỏ | Trung bình | Lớn |
|---|--:|--:|--:|
| Tích chập, theo batch | batch 1: 2,95× | batch 64: 40,30× | batch 1024: 56,09× |
| Kết nối đầy đủ, theo n | n = 256: **0,51×** (GPU thua) | n = 512: 1,10× | n = 4096: 8,37× |

Cùng xu hướng với bài báo: tăng tốc tăng theo batch và theo n. Tỉ lệ phụ thuộc loại lớp và kích thước,
không phải một hằng số 40 lần.

### 4. Strong scaling (T7)

3 epoch, 3 lần chạy ([chi tiết](results/t7_scaling.md)).

| Phương pháp | p | Thời gian (s) | Tăng tốc | Hiệu suất | Chờ/truyền |
|---|--:|--:|--:|--:|--:|
| C tuần tự | 1 | 6,900 | 1,00 | – | – |
| MPI đồng bộ | 2 | 3,673 | 1,88 | 93,9% | 2,2% |
| MPI đồng bộ | 4 | 2,111 | 3,27 | 81,7% | 13,5% |
| MPI đồng bộ | 6 | 1,932 | 3,57 | 59,5% | 27,8% |
| **CUDA** | GPU | **0,209** | **33,05** | – | – |

### Đối chiếu dự đoán

Dự đoán nằm trong [PLAN.md](PLAN.md), được ghi trước mỗi lần đo. Lịch sử commit đã được gộp trước khi mở PR
nên **git không chứng minh thứ tự thời gian này**. T1, T6, T7 và một phần T4 không độc lập vì đã có lần chạy thử trước đó.
T4 đối chứng và T8 được thiết kế sau khi xem T1, T4.

| Mã | Dự đoán | Kết quả |
|---|---|---|
| T1a | CUDA nhanh hơn 25–40 lần tới cùng accuracy | Đúng: 32,9 |
| T1b | MPI đồng bộ nhanh hơn 3,5–5,5 lần | **Sai**: 2,62 |
| T1c | Hai cách còn lại kém đồng bộ ít nhất 1,5 lần | Chưa chắc: trung bình tham số chậm hơn ở 5/5 seed nhưng 2 seed dưới 1,5 lần; bất đồng bộ bị giới hạn 12 epoch |
| T1d | Test 84–87% với mọi cách | **Sai** với bất đồng bộ: 83,8% |
| T4a | d = 15 làm mất trên 20 điểm ở bước lớn nhất | Đúng: 81,2 xuống 9,8 |
| T4b | Trong lưới 5 epoch có cấu hình d = 15 trên 70% | **Sai**: cao nhất 45,5 |
| T4c | Momentum chịu trễ kém hơn | Chưa kết luận: chênh lệch nhỏ so với dao động |
| T4d | d = 15, bước nhỏ, 15 epoch: trên 70% | Đúng: 77,7 và 74,8 (3 seed) |
| T4e | d = 31: tích 0,155 học được, 0,62 thì không | Đúng: 73,2 và 9,7 (3 seed) |
| T4f | d = 15: bước 0,01 nhiều seed học được hơn 0,02 | Đúng, nhưng chỉ hơn một seed |
| T4g | Không momentum tốt hơn ở ít nhất 8/10 seed | **Sai**: 7/10 |
| T4h | Tích 0,30–0,35: ít nhất 8/10 seed học được; 0,55–0,75: nhiều nhất 2/10 | **Sai**: 7/10 và 6/10; ô 0,55 là 4/10 |
| T6a | Conv batch từ 64: 30–60 lần | Đúng: 40–56 |
| T6b | Dense: GPU thua ở 256, thắng ở 1024 | Đúng: 0,51 và 2,55 |
| T7a | Tăng tốc trong khoảng dự đoán ở 2, 4, 6 rank | Đúng: 1,88; 3,27; 3,57 |
| T7b | Chờ/truyền tăng theo p, dưới 30% ở 6 rank | Đúng: 27,8% |
| T8a | Cùng momentum, đồng bộ nhanh hơn ở ít nhất 4/5 seed | Đúng: 5/5 ở cả hai mức (không đạt tính là chậm hơn) |
| T8b | Cho 20 epoch, bất đồng bộ đạt 85% ở ít nhất 4/5 seed | Đúng: 5/5 (không momentum) |

[T1](results/t1_time_to_accuracy.md) là lần so đầu, được T8 thay thế vì T1 so lệch momentum, learning rate
được chọn ở mép lưới và chỉ cho 12 epoch. Khi nới lưới, learning rate được chọn không đổi (0,01 và 0,05).
Lý do bản bất đồng bộ chỉ đạt 1/5 seed ở T1 chủ yếu là thiếu epoch: nó cần 12–17.

### Đã chứng minh và chưa chứng minh

Trong phạm vi máy, mô hình và dữ liệu này:

| Đã cho thấy | Chưa chứng minh |
|---|---|
| Lan truyền ngược đúng: lệch sai phân hữu hạn tối đa 8,9e-7, CUDA lệch C 1,8e-7 | Đúng trên Linux hoặc GPU khác |
| CUDA nhanh hơn tuần tự khoảng 33 lần với accuracy tương đương (5 seed) | Tỉ lệ này khi CPU dùng BLAS |
| Đồng bộ tới 85% nhanh hơn bất đồng bộ ở cả hai mức momentum, và nhanh hơn trung bình tham số khi cùng momentum 0,9 (5/5 seed) | Thứ tự này qua mạng giữa nhiều máy |
| Độ trễ nhân với bước càng lớn thì càng dễ không học được | Một ngưỡng chính xác, hay đúng cho mô hình khác |
| Trễ lớn (d = 15) vẫn học được khi bước nhỏ và đủ epoch | Momentum có làm chịu trễ kém hơn hay không |

## Giới hạn

- **Một máy:** MPI đi qua bộ nhớ chia sẻ, không suy ra được hành vi qua mạng.
- **Dữ liệu và mô hình nhỏ:** accuracy dừng quanh 86%. Mục tiêu là đo song song, không phải accuracy tối đa.
- **Ít seed:** learning rate chọn bằng 2 seed, đánh giá bằng 5. T8 dùng lại seed 11–15 của T1.
  Learning rate của trung bình tham số lấy từ T1 (nằm giữa lưới), chưa thử cách này khi không momentum.
- **Cùng một tập validation** vừa để chọn learning rate, vừa để đo ngưỡng 85%. Tập test đã được xem trong
  các lần chạy thử đầu.
- **Thời gian tới ngưỡng nhạy với nhiễu:** đồng bộ và tuần tự tương đương toán học nhưng epoch chạm 85% vẫn
  lệch tới 2, nên cần đọc kèm cột tăng tốc mỗi epoch.
- **Kết quả bản bất đồng bộ phụ thuộc tải máy**, vì thứ tự gradient về server thay đổi theo thời gian chạy.
- **T4 là mô phỏng độ trễ cố định** trên CNN không có BatchNorm, chưa chứng minh cho mô hình khác.
- **Không cài ADMM có biến đối ngẫu.** "ADMM.SGD" trong bài báo thực chất là SGD đồng bộ qua parameter server,
  tương ứng với `--mode sync`.
- **Mọi rank giữ một bản sao dữ liệu**, nên chưa đo chi phí ShuffleSGD.
- CPU không dùng BLAS. So với thư viện tối ưu thì tỉ lệ GPU/CPU có thể thấp hơn.
