# Kế hoạch và dự đoán

Dự đoán của mỗi phần được ghi trước khi chạy phần đó, phần bổ sung được ghi nối tiếp sau khi xem kết quả trước.
Bảng câu hỏi ở đầu được sửa lại sau. Lịch sử commit đã gộp nên git không chứng minh thứ tự thời gian.

Máy: Intel Core i7-13620H (6 lõi P, 4 lõi E), 32 GB DDR5-5600, RTX 4060 Laptop 8 GB, Windows 11.
Nguyên tắc: chỉ chạy code của thư mục này, chạy vừa phải (tối đa 6 lõi CPU), không đụng thư mục của thành viên khác.

| Mã | Câu hỏi |
|---|---|
| T1 | Cách nào nhanh nhất để tới **cùng accuracy**? |
| T4 | Khi gradient trễ, giảm learning rate có hết phân kỳ không? |
| T6 | Từng lớp: GPU hơn CPU bao nhiêu trên cùng một laptop? |
| T7 | MPI đồng bộ và CUDA tăng tốc bao nhiêu trên cùng một mã? |
| T8 | Đồng bộ và bất đồng bộ, so cùng momentum, cách nào nhanh hơn? |

Các mã T2, T3, T5 không dùng: đã bỏ trước khi chạy để phạm vi chỉ gồm code của thư mục này.

## Điều đã biết trước khi viết kế hoạch

Các lần chạy thử trước đó với CNN của thư mục này (Fashion-MNIST, 8.000 ảnh train) đã cho biết:
accuracy dừng ở khoảng 86%; CUDA nhanh hơn tuần tự khoảng 33 lần; lấy trung bình và bất đồng bộ cần
khoảng 8 epoch để đạt 85% trong khi đồng bộ cần 3; momentum 0,9 cộng gradient trễ từng gây phân kỳ.
Vì vậy dự đoán của T1 và một phần T4 **không độc lập** với các quan sát này. Các lần thử đó cũng đã
nhìn tập test, nên T1 dùng seed mới và tập validation riêng.

## T1. Thời gian tới cùng accuracy

- Dữ liệu: 7.000 train, 1.000 validation (ảnh 7000–7999), 2.000 test.
- Chọn learning rate bằng seed 101, 102 trên validation: ít epoch nhất để đạt 85% (không đạt tính là 13),
  hòa thì lấy validation cao hơn. Lưới: sync và avg {0,01; 0,02; 0,05}, async {0,05; 0,1; 0,2}.
- Đánh giá: seed 11–15, 12 epoch. Test lấy ở epoch có validation cao nhất.
- Thước đo: thời gian tới khi validation đạt 85% lần đầu, không tính thời gian đánh giá.

| Mã | Dự đoán |
|---|---|
| T1a | CUDA nhanh hơn tuần tự 25–40 lần tới cùng accuracy |
| T1b | MPI đồng bộ 6 rank nhanh hơn tuần tự 3,5–5,5 lần |
| T1c | Trung bình tham số và bất đồng bộ kém đồng bộ ít nhất 1,5 lần về thời gian tới 85% |
| T1d | Test trung bình ở epoch được chọn: 84–87% với mọi cách |

## T4. Staleness và learning rate

Mô phỏng tất định trên GPU: gradient tính ở bước t được áp ở bước t + d, tương đương W = d + 1 worker
chạy xoay vòng. d thuộc {0, 3, 7, 11, 15}; momentum {0; 0,9}; bước hiệu dụng lr/(1 − momentum) thuộc
{0,02; 0,05; 0,1; 0,2}; seed 11–13; 5 epoch; đánh giá trên validation.

| Mã | Dự đoán |
|---|---|
| T4a | Không momentum: ở d = 15, bước lớn nhất kém d = 0 trên 20 điểm |
| T4b | Ở d = 15, giảm bước tìm được cấu hình trên 70%, tức phân kỳ không phải tất yếu |
| T4c | Cùng bước hiệu dụng, momentum 0,9 chịu trễ kém hơn momentum 0 ở d từ 7 trở lên |

Đây là CNN nhỏ, không có BatchNorm, nên chỉ kiểm chứng cơ chế, không thay được phép đo VGG-16 của vinhvn.

## T4 bổ sung (ghi sau khi xem lưới T4, trước khi chạy phần bổ sung)

Lưới T4 bác bỏ T4b: ở d = 15 không cấu hình nào trên 70%. Các ô phân kỳ gợi ý ngưỡng theo tích
d x bước hiệu dụng, trong khoảng 0,35-0,55 (ô 7 x 0,05 còn học được, ô 11 x 0,05 thì không).

- T4d: ở d = 15, bước 0,01 (tích 0,15) và 0,005 (tích 0,075) với momentum 0 đạt trên 70% sau 15 epoch.
- T4e: d = 31 với bước 0,005 (tích 0,155) cũng trên 70%; d = 31 với bước 0,02 (tích 0,62) dưới 30%.

## T6, T7 (ghi trước khi chạy script mới)

Không độc lập: các lần chạy thử trước đã cho conv nhanh hơn tối đa khoảng 55 lần, dense hòa vốn quanh
n = 512, MPI đồng bộ khoảng 3,5 lần ở 4 rank và 4 lần ở 6 rank. Lần này đo lại bằng script có lặp.

- T6a: conv, batch từ 64: GPU nhanh hơn CPU một luồng 30-60 lần.
- T6b: dense một vector: GPU thua ở n = 256 và thắng ở n = 1024.
- T7a: speedup MPI đồng bộ ở 2 rank là 1,7-2,0; ở 4 rank là 2,8-3,7; ở 6 rank là 3,3-4,5.
- T7b: tỉ lệ chờ/truyền tăng theo số rank và dưới 30% ở 6 rank.

## T4 đối chứng và T8 (ghi sau khi xem T1, T4; trước khi chạy hai phép đo này)

Lý do: phần T4 bổ sung đổi cả bước lẫn số epoch ở d = 15, và T1 so bất đồng bộ không momentum với đồng bộ
có momentum, learning rate của cả hai nằm ở mép lưới.

T4 đối chứng: d = 15, 15 epoch, seed 11-20, bước {0,02; 0,01; 0,005}, momentum {0; 0,9}.

- T4f: với cùng 15 epoch, bước 0,01 có nhiều seed trên 70% hơn bước 0,02 (momentum 0).
- T4g: momentum 0 tốt hơn momentum 0,9 ở ít nhất 8/10 seed tại bước 0,02.

T8: bốn biến thể {đồng bộ, bất đồng bộ} x {momentum 0; 0,9}, lưới learning rate tự nới tới khi điểm tốt
nhất không ở mép, tối đa 20 epoch, seed đánh giá 11-15. Trung bình tham số giữ lr 0,02 vì đã ở giữa lưới.
Hệ số 1/căn(số worker) ở server chỉ là hằng số nhân, được bù khi dò learning rate.

- T8a: cùng momentum, đồng bộ tới 85% nhanh hơn bất đồng bộ ở ít nhất 4/5 seed (không đạt tính là chậm hơn).
- T8b: sau khi nới lưới và cho 20 epoch, biến thể bất đồng bộ tốt hơn đạt 85% ở ít nhất 4/5 seed.

## T4 đối chứng vùng ngưỡng (ghi sau T4f/T4g, trước khi chạy)

T4 đối chứng cho thấy ô tích 0,30 (d = 15) học được ở 9/10 seed khi đủ 15 epoch, nên ngưỡng đo ở 5 epoch
có thể bị lẫn với tốc độ học.

- T4h: với 15 epoch, 10 seed, momentum 0: các ô tích 0,30-0,35 học được (trên 70%) ở ít nhất 8/10 seed,
  các ô tích 0,55-0,75 học được ở nhiều nhất 2/10 seed.

## T8 chạy bù (ghi trước khi chạy)

Ở T8, bất đồng bộ momentum 0,9 có một seed chưa đạt 85% sau 20 epoch (cao nhất 84,1%). Chạy lại cả 5 seed
với tối đa 40 epoch, đúng một lượt, báo cáo kết quả dù đạt hay không; đồng bộ chạy xen kẽ làm đối chứng.
