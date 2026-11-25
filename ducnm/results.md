# Tổng hợp kết quả thực nghiệm HPC (`ducnm/`)

- **Môi trường đo thực tế**: Python 3.12.2, NumPy 2.4.4, OS `Windows-10-10.0.19045-SP0`, CPU `Intel64 Family 6 Model 94 Stepping 3, GenuineIntel`
- **Trạng thái GPU thực**: `PyTorch unavailable in current Python environment` (phần GPU & mạng phân tán dùng **Mock/Simulated Output** có ghi rõ nguồn `mock_simulation` / `analytical_alpha_beta_model`)
- **Kiểm tra đúng đắn (Gradient Check)**: sai số tương đối lớn nhất (sai phân hữu hạn `eps=1e-6`) = `3.008e-06` (đạt yêu cầu `< 1e-4`).

---

## Bài toán 1: Trong CNN, phép nhân ma trận là bottleneck

Mô hình kiểm thử: **SmallCNN** (`1×28×28 → Conv1(8@5×5) → Pool → Conv2(16@5×5) → Pool → Dense(64) → Dense(10)`), **54,314 tham số** (`217,256 bytes` FP32).

1. **Vì sao dùng `im2col + GEMM` thay vì vòng lặp trực tiếp (`batch=4`)**:
   - Direct 7 vòng lặp lồng nhau: **475.06 ms**
   - `im2col + GEMM`: **0.50 ms** → nhanh hơn **945.4×**, sai khác tối đa `2.86e-06`.

2. **Phân rã FLOP & thời gian thực tế trên CPU (`real_cpu_measurement`)**:

| Batch | GEMM FLOPs (%) | Tổng step CPU (ms) | GEMM CPU (ms) | GEMM (%) | im2col/col2im (%) | ReLU/Pool/SGD (%) |
|--:|--:|--:|--:|--:|--:|--:|
| 16 | 98.77% | 15.65 | 4.10 | 26.2% | 28.2% | 45.6% |
| 64 | 98.98% | 85.45 | 42.51 | 49.8% | 19.8% | 30.5% |
| 128 | 99.01% | 181.36 | 88.24 | 48.7% | 21.2% | 30.1% |
| 256 | 99.03% | 239.97 | 71.25 | 29.7% | 31.1% | 39.2% |

   > *Ghi chú NumPy CPU*: Ở mô hình nhỏ (`8→16` kênh), OpenBLAS chạy GEMM rất nhanh nên chi phí cấp phát mảng Python cho `im2col`/`MaxPool` lộ rõ (~30–45%). Khi kiểm thử lớp Conv rộng hơn thực tế (`Conv 32->64 (3x3, 28x28, batch=64)`), GEMM chiếm **99.88% FLOPs** và **50.1% thời gian CPU thực** (`90.88 / 181.51 ms`), còn `im2col/col2im` chỉ `41.6%` và `elementwise` chỉ `8.4%`.

3. **Mock GPU Kernel Time Share (`mock_simulation`, mô phỏng Tesla T4 Roofline)**:

| Batch | Tổng Kernel (ms) | GEMM Kernel (ms) | GEMM (%) | Non-GEMM BW-bound (%) |
|--:|--:|--:|--:|--:|
| 16 | 0.4729 | 0.3989 | 84.3% | 15.7% |
| 64 | 0.9325 | 0.8255 | 88.5% | 11.5% |
| 128 | 1.3741 | 1.2230 | 89.0% | 11.0% |
| 256 | 2.1706 | 1.9312 | 89.0% | 11.0% |

> **Kết luận Bài toán 1**: Trong cả lý thuyết FLOP (~98,8–99,9% là GEMM) lẫn mô phỏng GPU (~84–89% thời gian kernel), **phép nhân ma trận (GEMM)** chiếm phần chi phí áp đảo. Trên CPU với `im2col + GEMM`, riêng cụm `im2col + GEMM` chiếm ~54–91% thời gian tùy độ rộng kênh. Tối ưu GEMM (tiling, shared memory, cuBLAS/Tensor Core) vì thế quyết định trực tiếp tốc độ huấn luyện CNN.

---

## Bài toán 2: GPU mạnh hơn CPU trong huấn luyện

1. **Huấn luyện CNN 5 epoch (`2,048` ảnh train, `512` ảnh val, `batch=64`)**:

| Thiết bị | Nguồn số liệu | Thời gian 5 epoch | TB / epoch | Val Accuracy | Tăng tốc |
|---|---|--:|--:|--:|--:|
| CPU (NumPy) | `real_cpu_measurement` | 18.561 s | 3712.3 ms | 100.0% | 1.0× |
| GPU T4 (chỉ kernel) | `mock_simulation` | 0.1431 s | 28.63 ms | 100.0% | **129.7×** |
| GPU T4 (kèm H2D PCIe) | `mock_simulation` | 0.1504 s | 30.07 ms | 100.0% | **123.5×** |

2. **Lớp Convolution chuẩn bài báo (`224×224, 96@11×11, stride=4`) — CPU thật vs GPU mô phỏng**:

| Batch | CPU thật (ms) | GPU kernel mock (ms) | Speedup (kernel) | GPU + PCIe mock (ms) | Speedup (+PCIe) |
|--:|--:|--:|--:|--:|--:|
| 1 | 1.92 | 0.0930 | 20.7× | 0.2550 | 7.5× |
| 8 | 14.78 | 0.5690 | 26.0× | 1.6554 | 8.9× |
| 16 | 28.34 | 1.1130 | 25.5× | 3.2557 | 8.7× |
| 32 | 62.25 | 2.2010 | 28.3× | 6.4564 | 9.6× |
| 64 | 137.90 | 4.3770 | 31.5× | 12.8579 | 10.7× |
| 128 | 353.85 | 8.7290 | 40.5× | 25.6607 | 13.8× |

3. **Matvec (giới hạn băng thông) — Chứng minh GPU KHÔNG thắng ở kích thước quá nhỏ**:

| Kích thước `n` | CPU thật (ms) | GPU mock (ms) | Speedup | Nhận xét |
|--:|--:|--:|--:|---|
| 256 | 0.0175 | 0.0260 | 0.68× | GPU slower at tiny n due to 25us kernel launch overhead |
| 512 | 0.0364 | 0.0289 | 1.26× | GPU BW wins |
| 1024 | 0.1931 | 0.0407 | 4.74× | GPU BW wins |
| 2048 | 0.8929 | 0.0879 | 10.16× | GPU BW wins |
| 4096 | 2.9082 | 0.2765 | 10.52× | GPU BW wins |

> **Kết luận Bài toán 2**: Trên cấu hình đo và mô phỏng T4, GPU nhanh hơn CPU **129.7×** (chỉ kernel) hoặc **123.5×** (gồm PCIe H2D) ở huấn luyện SmallCNN, và nhanh hơn **21–42×** ở lớp Conv `224×224` (chỉ kernel) nhờ hàng nghìn nhân song song và băng thông VRAM cao (~320 GB/s so với ~20–30 GB/s trên CPU). Tuy nhiên ở kích thước quá nhỏ (`n=256`), độ trễ launch kernel (~25 µs) khiến GPU thua CPU (`0.68×`).

---

## Bài toán 3: Phân tích Communication Cost trong Gradient Descent

Áp dụng mô hình Hockney $\alpha$–$\beta$: $T_{\text{p2p}}(M) = \alpha + M / B$.

1. **So sánh 4 thuật toán Allreduce trên PCIe (`alpha=15 us, BW=10 GB/s`, `mock/analytical`)**:

| Workers (`P`) | Gradient (`M`) | Naive (us) | Tree (us) | Recursive Doubling (us) | Ring (us) | Nhanh nhất |
|--:|--:|--:|--:|--:|--:|---|
| 4 | 2 KB | 91.2 | 60.8 | 30.4 | 90.3 | `rd` |
| 4 | 212 KB | 220.4 | 146.9 | 73.5 | 122.6 | `rd` |
| 4 | 1.6 MB | 1066.9 | 711.3 | 355.6 | 334.2 | `ring` |
| 4 | 42.6 MB | 26907.5 | 17938.3 | 8969.2 | 6794.4 | `ring` |
| 8 | 2 KB | 212.9 | 91.2 | 45.6 | 210.4 | `rd` |
| 8 | 212 KB | 514.2 | 220.4 | 110.2 | 248.0 | `rd` |
| 8 | 1.6 MB | 2489.5 | 1066.9 | 533.5 | 494.9 | `ring` |
| 8 | 42.6 MB | 62784.2 | 26907.5 | 13453.8 | 8031.8 | `ring` |

2. **Strong Scaling theo Global Batch Size (`P=1,2,4,8`, gradient `1.63 MB`, Ring Allreduce trên PCIe)**:

| Global Batch | Workers (`P`) | Compute (ms) | Comm (ms) | Comm Share (%) | Speedup vs 1 GPU | Efficiency (%) |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 1 | 0.235 | 0.000 | 0.0% | 1.00× | 100.0% |
| 64 | 2 | 0.178 | 0.193 | 52.0% | 0.64× | 31.7% |
| 64 | 4 | 0.149 | 0.334 | 69.2% | 0.49× | 12.2% |
| 64 | 8 | 0.134 | 0.495 | 78.6% | 0.37× | 4.7% |
| 256 | 1 | 0.581 | 0.000 | 0.0% | 1.00× | 100.0% |
| 256 | 2 | 0.350 | 0.193 | 35.5% | 1.07× | 53.5% |
| 256 | 4 | 0.235 | 0.334 | 58.7% | 1.02× | 25.5% |
| 256 | 8 | 0.178 | 0.495 | 73.6% | 0.86× | 10.8% |
| 1024 | 1 | 1.963 | 0.000 | 0.0% | 1.00× | 100.0% |
| 1024 | 2 | 1.042 | 0.193 | 15.6% | 1.59× | 79.5% |
| 1024 | 4 | 0.581 | 0.334 | 36.5% | 2.15× | 53.6% |
| 1024 | 8 | 0.350 | 0.495 | 58.5% | 2.32× | 29.0% |
| 4096 | 1 | 7.493 | 0.000 | 0.0% | 1.00× | 100.0% |
| 4096 | 2 | 3.806 | 0.193 | 4.8% | 1.87× | 93.7% |
| 4096 | 4 | 1.963 | 0.334 | 14.6% | 3.26× | 81.5% |
| 4096 | 8 | 1.042 | 0.495 | 32.2% | 4.88× | 61.0% |

3. **Đánh đổi giữa các chiến lược SGD phân tán (`SmallCNN 54K params`, 10 epoch, mạng 10GbE, `P=4`)**:

| Chiến lược | Số lần sync (10 ep) | Tổng lượng truyền (MB) | Tổng thời gian truyền (ms) | Staleness / Rủi ro hội tụ |
|---|--:|--:|--:|---|
| Sync SGD (allreduce every step) | 400 | 130.35 | 224.28 | 0 — Lowest (mathematically equivalent to single-worker batch SGD) |
| Parallel SGD (avg every 10 steps) | 40 | 13.04 | 22.43 | 0-9 local steps drift — Low on IID; requires ShuffleSGD on non-IID |
| Parallel SGD (avg once per epoch) | 10 | 3.26 | 5.61 | Up to 40 steps drift — High on non-IID data without shuffling |
| Async Downpour (PS, k* = T/tau = 3.4 workers) | 400 | 173.8 | 134.28 | ~W-1 = 3 steps — Degrades if staleness x lr is large; saturates above k* |

> **Kết luận Bài toán 3**: Chi phí truyền thông `T_comm` phụ thuộc kích thước gradient `M` (không phụ thuộc batch size), trong khi phần tính toán tỷ lệ thuận với batch. Do đó ở `batch=64`, truyền thông chiếm **52,0–78,6%** thời gian bước (`P=2..8`) và **làm chậm hơn chạy 1 GPU** (`0,64× → 0,37×`); chỉ khi tăng global batch lên `1024–4096` thì tỷ lệ truyền thông mới giảm xuống **4,8–32,2%** để multi-GPU đạt speedup thực sự (`1,87× → 4,88×`).
