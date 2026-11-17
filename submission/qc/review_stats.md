# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 525; đối tượng sau duyệt: 494; ngưỡng accepted: IoU >= 0.9
- Đã **loại 25 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 455 | 455/525 = 86.7% | 525 box máy |
| edited (sửa) | 12 | 12/525 = 2.3% | 525 box máy |
| deleted (xoá) | 58 | 58/525 = 11.0% | 525 box máy |
| added (vẽ thêm) | 27 | 27/494 = 5.5% | 494 đối tượng sau duyệt |

Trong số deleted + added, **8 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 19.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 249 | 2 | 11 | 9 |
| bicycle | 6 | 0 | 3 | 2 |
| motorbike | 119 | 4 | 15 | 5 |
| car | 67 | 0 | 10 | 4 |
| bus | 3 | 0 | 1 | 0 |
| truck | 1 | 0 | 2 | 1 |
| three_wheeler | 10 | 6 | 16 | 6 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 73 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 45 | 27 | 28 | 62.5% (KTC 95%: 51–73%) | 61.6% (KTC 95%: 50–72%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 7/12 = 58.3% (KTC 95%: 32–81%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 1/3
  - loose: 2/3
  - missing: 1/3
  - extra: 3/3
  - lọt: img_026.jpg (wrong_class), img_005.jpg (missing), img_018.jpg (loose), img_025.jpg (wrong_class), img_021.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
