# Kế hoạch và dự đoán trước thí nghiệm (`ducnm/`)

Tài liệu ghi nhận các giả thuyết kiểm chứng cho 3 câu hỏi trọng tâm môn Tính toán hiệu năng cao, tham khảo phương pháp từ `hieunv/`, `ltnaphixay/`, `thinhnd/`, `tuannn/`, `vinhvn/`.

| Mã | Bài toán | Giả thuyết / Dự đoán | Cách kiểm chứng |
|---|---|---|---|
| H1a | CNN bottleneck | Chuyển Conv sang `im2col + GEMM` nhanh hơn vòng lặp trực tiếp hàng trăm đến hàng nghìn lần mà sai số số học `< 1e-4` | So `conv_forward_direct_naive` với `im2col + GEMM` ở `batch=4` |
| H1b | CNN bottleneck | Trong huấn luyện CNN, các phép nhân ma trận (`Conv GEMM + FC GEMM`) chiếm `> 95%` tổng số FLOP và chiếm đa số thời gian kernel GPU | Đếm FLOP từng lớp + đo thời gian thực trên CPU + mô hình hóa GPU kernel |
| H2a | GPU vs CPU | GPU nhanh hơn CPU hàng chục lần trong huấn luyện CNN và lớp Conv `224×224` khi `batch` đủ lớn; chênh lệch giảm khi tính thêm chi phí chép PCIe H2D | Huấn luyện `SmallCNN` 5 epoch + benchmark Conv `224×224` |
| H2b | GPU vs CPU | Ở kích thước ma trận rất nhỏ (`n = 256`), GPU có thể **chậm hơn CPU** do độ trễ khởi chạy kernel (~25 µs) lớn hơn thời gian tính toán | Quét `matvec` từ `n = 256` đến `4096` |
| H3a | Communication cost | Ở thông điệp nhỏ, `recursive doubling` (`O(log P)`) nhanh hơn `ring` (`O(P)` latency); ở thông điệp lớn, `ring` nhanh nhất nhờ truyền băng thông `2(P-1)/P · M/B` | Mô hình Hockney α–β cho `naive`, `tree`, `rd`, `ring` |
| H3b | Communication cost | Với batch nhỏ (`64`), thêm GPU làm chậm huấn luyện vì `T_comm` cố định theo kích thước gradient `M`; tăng global batch (`1024–4096`) làm giảm `comm_share` và mở khóa speedup | Quét `global_batch = 64..4096` với `P = 1, 2, 4, 8` |
| H3c | Communication cost | Giảm tần suất đồng bộ (Parallel SGD) giảm mạnh tổng lượng truyền nhưng đánh đổi nguy cơ trôi tham số (staleness) trên dữ liệu không IID | So Sync SGD, Parallel SGD (H=10, epoch), Async Downpour |
