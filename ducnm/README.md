# Thực nghiệm HPC nền tảng: Bottleneck GEMM trong CNN, CPU vs GPU và Communication Cost (`ducnm/`)

Thư mục `ducnm/` hoàn thiện mã nguồn chạy được bằng **Python + NumPy** (tự động dùng PyTorch CUDA nếu có, và sinh **Mock/Simulated Output** có ghi rõ nguồn cho phần GPU & mạng phân tán khi chạy trên máy không có GPU CUDA / cụm nhiều node), phục vụ chứng minh 3 bài toán cốt lõi của môn Tính toán hiệu năng cao:

1. **Trong CNN, phép nhân ma trận (GEMM) là bottleneck.**
2. **GPU mạnh hơn CPU trong huấn luyện (và khi nào GPU không thắng).**
3. **Phân tích communication cost trong gradient descent phân tán.**

Các tham số kiến trúc và mô hình phân tích tham khảo từ các thư mục khác trong repo (`thinhnd/`, `ltnaphixay/`, `hieunv/`, `tuannn/`, `vinhvn/`), **không chỉnh sửa bất kỳ thư mục nào khác**.

---

## 1. Cấu trúc thư mục

```text
ducnm/
├── README.md                  Tài liệu hướng dẫn, phương pháp và tổng hợp
├── PLAN.md                    Giả thuyết và dự đoán trước khi chạy
├── cnn_core.py                CNN 5 lớp tự viết (im2col+GEMM, đếm FLOP, timer, gradient check)
├── exp1_cnn_bottleneck.py     Thí nghiệm 1: chứng minh GEMM là bottleneck trong CNN
├── exp2_cpu_vs_gpu.py         Thí nghiệm 2: so sánh CPU (thực) vs GPU (CUDA hoặc mock)
├── exp3_comm_cost.py          Thí nghiệm 3: mô hình α–β, scaling theo batch và chiến lược SGD
├── run_all.py                 Chạy kiểm tra + cả 3 thí nghiệm, sinh results.json & results.md
├── train_demo.py              Demo hồi quy tuyến tính nền tảng ban đầu (vẫn chạy độc lập)
├── outputs/                   JSON chi tiết từng thí nghiệm (+ train_demo.json)
├── results.json               Dữ liệu tổng hợp toàn bộ thí nghiệm
└── results.md                 Báo cáo kết quả chi tiết (phân tách rõ đo thực vs mock)
```

---

## 2. Yêu cầu & Cách chạy

Yêu cầu tối thiểu: **Python 3.10+**, **NumPy** (không bắt buộc PyTorch/CUDA hay trình biên dịch C).

```powershell
cd ducnm

# 1. Kiểm tra tính đúng đắn của lan truyền ngược CNN (finite-difference gradient check)
python cnn_core.py

# 2. Chạy từng thí nghiệm riêng lẻ
python exp1_cnn_bottleneck.py
python exp2_cpu_vs_gpu.py
python exp3_comm_cost.py

# 3. Chạy toàn bộ và tự động cập nhật results.json, results.md, outputs/*.json
python run_all.py

# (Tuỳ chọn) Chạy lại bài demo hồi quy tuyến tính ban đầu
python train_demo.py
```

---

## 3. Tóm tắt phương pháp và phân định Đo thực vs Mock Output

| Thành phần | Dữ liệu thực (`real_cpu_measurement`) | Dữ liệu mô phỏng (`mock_simulation` / `analytical`) |
|---|---|---|
| **Bài toán 1 (CNN Bottleneck)** | Đếm FLOP chính xác; đo thời gian trực tiếp `direct loop` vs `im2col+GEMM`; đo thời gian từng pha của `SmallCNN` (54.314 tham số) và lớp rộng `Conv 32→64` trên CPU | Mô phỏng thời gian kernel GPU (Tesla T4 roofline: GEMM compute-bound + elementwise BW-bound) |
| **Bài toán 2 (GPU vs CPU)** | Huấn luyện `SmallCNN` 5 epoch (`2,048` ảnh) trên CPU; đo lớp `Conv 224×224 (96@11×11/s4)` và `Matvec/GEMM` trên CPU | Nếu không có PyTorch+CUDA, mô phỏng GPU T4 gồm độ trễ launch (~25 µs), compute/BW và chi phí copy PCIe H2D/D2H |
| **Bài toán 3 (Communication Cost)** | Kiểm chứng công thức theo tham số mô hình `SmallCNN` (`217 KB`) và `MLP` (`1.63 MB`) | Mô hình Hockney $\alpha$–$\beta$ cho `naive`, `tree`, `rd`, `ring` trên NVLink / PCIe / 10GbE; quét `global_batch = 64..4096` |

Xem kết quả đầy đủ mới nhất tại [results.md](results.md) và dữ liệu gốc tại [results.json](results.json).
