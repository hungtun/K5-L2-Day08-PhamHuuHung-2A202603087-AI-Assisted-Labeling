# Báo cáo Day 08 — 2A202603087 - Phạm Hữu Hưng

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv` & `qc/threshold_sweep.md`) |
|---|---|---|---|
| `car` | YOLO26m (`yolo26m.pt`) | **`0.15`** | Từ `r05` (`y26_all`): tại ngưỡng `0.15` đạt mục tiêu Recall **100%** (8/8 vật gold, $\ge 90\%$), Precision **40.0%** (8/20 box), F1 = `0.571`. |
| `person`, `motorbike` | YOLO26m (`yolo26m.pt`) | **`0.30`** *(mặc định)* | Từ `r01`, `r02`, `r05`: không có ngưỡng nào $\le 0.50$ đạt Recall 90% (tối đa ở `0.10` chỉ đạt **76.5%** với 26/34 `person` và **54.5%** với 12/22 `motorbike` do nhiều người lái/xe máy bị che khuất); dùng ngưỡng mặc định `0.30` để cân bằng (tại `0.30`: `person` P=55.9%, R=55.9%; `motorbike` P=83.3%, R=45.5%). |
| `bicycle`, `bus`, `truck` | YOLO26m (`yolo26m.pt`) | **`0.30`** *(mặc định)* | Cỡ mẫu trong 4 ảnh gold $< 5$ đối tượng (`bicycle` = 2, `bus` = 0, `truck` = 0) nên không đủ cơ sở thống kê để chọn ngưỡng riêng; giữ mặc định `0.30`. |
| `three_wheeler` | YOLOE-26s (`yoloe-26s-seg.pt`, prompt: `'tuk-tuk'`, `'cyclo'`, `'cargo tricycle'`) | **`0.25`** | Từ `r03`, `r06` (`yoloe_all`): tại ngưỡng `0.25` đạt Recall **100%** (7/7 vật gold, $\ge 90\%$), Precision **87.5%** (7/8 box), F1 = `0.933`. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "car=0.15,three_wheeler=0.25" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Chọn **SAM 2.1 base (`sam2.1_b.pt`)** (`r04`). Lý do: chạy mượt trên GPU Apple MPS (`4.10s/ảnh`), chất lượng đường biên mask bám sát đối tượng tốt hơn bản nhẹ `mobile_sam.pt` (SAM 1), dung lượng vừa phải (~154 MB, không làm tràn RAM khi chạy song song với Docker CVAT) và không bị khóa quyền truy cập (*gated*) như SAM 3.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào?
  1. **Lỗi thiếu lớp trong từ vựng cố định (Closed-set taxonomy mismatch):** YOLO26 được huấn luyện trên 80 lớp COCO (không có `three_wheeler`), nên dù nâng hay hạ ngưỡng thì nó vẫn gọi tuk-tuk là `car` (`img_004.jpg`: `car 0.35`, `0.34`) hoặc gọi bánh sau xích lô là `bicycle` (`img_032.jpg`: `bicycle 0.74`). **Cách sửa:** Kết hợp thêm model open-vocabulary **YOLOE** với text prompt (`'tuk-tuk'`, `'cyclo'`, `'cargo tricycle'`) rồi dùng `merge_prelabels.py` (ưu tiên `yoloe` đứng trước, NMS IoU `0.6`) và kiểm tra lại bằng mắt khi review.
  2. **Lỗi hình học của box (Box không ôm hết vật hoặc trùng lặp):** Ở `img_032.jpg`, box `three_wheeler` của YOLOE chỉ ôm phần ghế khách phía trước mà thiếu khung xe đạp phía sau; ở `img_002.jpg`, YOLOE sinh 2 box lồng nhau trên cùng 1 chiếc tuk-tuk (`0.92` và `0.43`). Lỗi biên box phải được con người chỉnh tay (edit) trên CVAT theo đúng guideline v1.0.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - **Điểm cao mà sai:** Trên `img_032.jpg`, YOLO26 dự đoán phần khung sau của chiếc xích lô là `bicycle` với điểm tin cậy rất cao (`conf = 0.74`), trong khi theo guideline v1.0 cả chiếc xích lô phải thuộc lớp `three_wheeler`.
  - **Điểm thấp mà đúng:** Trên `img_032.jpg`, chiếc xích lô đỗ bên phải được YOLOE phát hiện đúng lớp `three_wheeler` nhưng chỉ đạt `conf = 0.32` (và chiếc bên trái đạt `conf = 0.40`); tương tự trên `img_007.jpg` và `img_011.jpg`, nhiều người đi bộ và xe máy ở trung cảnh/bị che khuất một phần được YOLO26 bắt đúng nhưng chỉ có `conf = 0.27–0.35` (bị mất hoàn toàn khi nâng ngưỡng lên `0.5` ở `r02`).
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: Tập gold chỉ có **4 ảnh (73 đối tượng)** và một số lớp có cỡ mẫu cực nhỏ (`bicycle` chỉ có 2 vật, `bus` và `truck` có 0 vật trong 4 ảnh gold). Việc vừa dùng 4 ảnh gold này để chọn ngưỡng (`threshold_sweep.py`) vừa dùng chính nó để báo cáo Precision/Recall/F1 của máy gây ra **thiên lệch tối ưu hóa trên tập kiểm tra (data leakage / overfitting trên tập validation nhỏ)**, làm cho điểm số của máy trên gold cao hơn (lạc quan hơn) so với hiệu năng thực tế trên 24 ảnh batch.

## 2. AI error profile (từ `qc/review_stats.md`)

*(Đã loại 25 đối tượng nằm quanh các bẫy nhân tạo để đánh giá khách quan chất lượng máy trên 24 ảnh batch; ngưỡng `accepted`: $\text{IoU} \ge 0.90$).*

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| `accepted` (giữ nguyên) | **455** | **86.7%** (`455/525`) | `525` box máy |
| `edited` (sửa kích thước/vị trí) | **12** | **2.3%** (`12/525`) | `525` box máy |
| `deleted` (xoá) | **58** | **11.0%** (`58/525`) | `525` box máy |
| `added` (vẽ thêm) | **27** | **5.5%** (`27/494`) | `494` đối tượng sau duyệt |

> Trong số `deleted` + `added`, có **8 trường hợp là đổi lớp** (máy đặt box đúng vị trí nhưng sai lớp, tính bằng 1 `deleted` + 1 `added`). Như vậy số đối tượng máy **bỏ sót thật sự** được người duyệt vẽ thêm là $27 - 8 = \mathbf{19}$ đối tượng (`19/494 = 3.8%`).

- **Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?**
  - **`three_wheeler`** là lớp có tỉ lệ sai lệch cao nhất: chỉ có **10/32 box máy được giữ nguyên (`31.3%`)**, phải sửa biên **6 box (`18.8%`)**, xoá **16 box (`50.0%`)** và vẽ thêm **6 box**. Nguyên nhân: YOLOE với prompt chữ (`'tuk-tuk'`, `'cyclo'`, `'cargo tricycle'`) ở ngưỡng `0.25` hay sinh 2 box chồng nhau trên cùng một xe (như một box ôm cả mui và một box chỉ ôm thân dưới) hoặc bắt nhầm xe máy chở hàng cồng kềnh / xe ba gác tự chế; đồng thời box của xích lô thường không ôm hết khung xe đạp phía sau nên phải kéo lại (`edited = 6`).
  - **`motorbike`** đứng thứ hai về số lượng lỗi tuyệt đối (`deleted = 15/138`, `edited = 4`, `added = 5`) do các ảnh giao thông dày đặc (`img_005.jpg`, `img_018.jpg`, `img_021.jpg`, `img_024.jpg`, `img_028.jpg`, `img_029.jpg`) có nhiều xe máy ở xa cao `< 20px` (máy vẫn vẽ từng chiếc trong khi guideline v1.0 yêu cầu xóa lẻ và gộp vào `ignore_region`) hoặc bị che khuất nặng.
- **So với gold:** Trên 4 ảnh gold (**73 đối tượng gold** làm mẫu số của Recall, **72 box máy** làm mẫu số của Precision):
  - **Precision của máy:** **62.5%** (`45/72` box máy, KTC 95%: `51%–73%`), với `TP = 45`, `FP = 27`.
  - **Recall của máy:** **61.6%** (`45/73` đối tượng gold, KTC 95%: `50%–72%`), với `TP = 45`, `FN = 28`.
  - *Lưu ý:* Đây đã là con số **lạc quan** vì ngưỡng từng lớp được chọn trực tiếp trên chính 4 ảnh gold này (*in-sample*).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Bắt bẫy:** **7/12 = 58.3%** (KTC 95%: `32%–81%`, thấp hơn ngưỡng mục tiêu `80%` = `10/12`).
  - Phân bố theo loại bẫy:
    - `extra` (box ma): **3/3 (100%)** — bắt đủ trên `img_008.jpg`, `img_017.jpg`, `img_022.jpg`.
    - `loose` (box lỏng/lệch): **2/3 (66.7%)** — bắt được trên `img_023.jpg`, `img_031.jpg`; **lọt 1** trên `img_018.jpg` (`motorbike` tại `[803, 536, 53, 44]` lệch nhẹ ~8 px so với box gốc `[812, 527, 45, 49]`).
    - `missing` (bỏ sót): **1/3 (33.3%)** — bắt được trên `img_003.jpg`; **lọt 2** trên `img_005.jpg` (`person` tại `[298, 231, 49, 81]`) và `img_021.jpg` (`person` ở sát mép phải `[948, 353, 75, 263]`).
    - `wrong_class` (sai lớp): **1/3 (33.3%)** — bắt được trên `img_013.jpg`; **lọt 2** trên `img_025.jpg` (`person` bị đổi thành `bicycle` tại `[708, 151, 70, 153]`) và `img_026.jpg` (`person` nhỏ ở xa `12×32 px` bị đổi thành `bicycle` tại `[750, 317, 12, 32]`).
  - **Vì sao lọt (đã bỏ bước nào của checklist)?**
    1. **Bỏ qua Bước 0 (Ẩn nhãn máy & quét theo góc phần tư trước) trên các ảnh đông đúc (`DENSE`):** Cả 3 bẫy lọt thuộc loại `missing` và `loose` (`img_005.jpg`, `img_018.jpg`, `img_021.jpg`) đều nằm trong nhóm 6 ảnh dày đặc nhất bộ dữ liệu ($> 50$ vật/ảnh). Khi mở ảnh có hàng chục box đè lên nhau, mắt bị hút theo các box sẵn có mà không ẩn nhãn để tự đếm người ở mép phải (`img_021.jpg`) hoặc trung cảnh (`img_005.jpg`).
    2. **Làm lướt Bước 3 (Kiểm tra nhãn lớp từng box) đối với cụm người–xe:** Ở `img_025.jpg` và `img_026.jpg`, khi thấy khung hình chữ nhật đã ôm đúng vị trí đối tượng, não tự động giả định nhãn lớp của máy cũng đúng mà không đọc kỹ nhãn chữ `bicycle` gắn trên box của `person`.
- **`accepted` cao (86.7%) mà bắt bẫy thấp (58.3%) nói gì về cách bạn duyệt?**
  - Đây là bằng chứng điển hình của **thiên kiến tự động hóa (automation bias) và hiệu ứng neo nhãn (anchoring)**: người duyệt rất giỏi phát hiện các lỗi "dương tính giả hiển nhiên" (`extra` bắt 3/3 vì nhìn vào box thấy nền trống), nhưng lại **quá tin vào sự vắng mặt của box** (lọt 2/3 bẫy `missing`) và **quá tin vào độ khít/nhãn mặc định của máy** trong cảnh đông đúc. Con số `accepted = 86.7%` cao một phần vì máy tốt ở các vật dễ, nhưng phần khác là do người duyệt **chấp nhận thụ động** các lỗi tinh vi mà lẽ ra phải sửa (`edited` chỉ 2.3% và `added` chỉ 5.5%, trong khi trên tập gold máy có Recall chỉ 61.6%, tức bỏ sót tới 38.4% đối tượng!).

## 4. Máy có đáng dùng không?

- **Thời gian (từ `time_log.csv`):**
  - Gán tay từ đầu (`gold_manual`): **9.25 phút/ảnh** (`37 phút / 4 ảnh`, `73 đối tượng` $\approx 30.4\text{ giây/đối tượng}$).
  - Duyệt có pre-label (`batch_review`): **1.46 phút/ảnh** (`35 phút / 24 ảnh`, `505 đối tượng` $\approx 4.2\text{ giây/đối tượng}$) — **nhanh gấp 6.3 lần** (tiết kiệm **84.2%** thời gian trên mỗi ảnh).
  - **Thiên lệch khi so sánh thời gian:** Khi bước vào duyệt 24 ảnh batch, người gán đã làm xong 4 ảnh gold trước đó nên đã thuộc lòng 8 lớp nhãn, phím tắt CVAT (`N`, `Del`, `F`, `Ctrl+S`) và quy tắc `< 20px` (hiệu ứng học tập — *learning curve*). Ngoài ra, tốc độ `1.46 phút/ảnh` đạt được một phần do duyệt hơi nhanh ở các ảnh đông (dẫn đến bắt bẫy chỉ đạt 58.3%); nếu tuân thủ nghiêm ngặt ~3 phút/ảnh để bắt $\ge 80\%$ bẫy thì thời gian thực tế sẽ khoảng `2.5–3.0 phút/ảnh` (vẫn tiết kiệm ~68–73% so với gán tay).
- **Kết luận:** **CÓ đáng dùng**, nhưng **phải kết hợp đúng model theo lớp và siết quy trình kiểm tra lỗi bỏ sót**:
  - Rất đáng dùng cho `car`, `person`, `motorbike` ở tiền cảnh/trung cảnh (`accepted` đạt `249/262 = 95.0%` với `person`, `67/77 = 87.0%` với `car`, `119/138 = 86.2%` với `motorbike`), giúp giảm mạnh thao tác vẽ box mới.
  - Với `three_wheeler` (`accepted` chỉ `10/32 = 31.3%`) và các cụm xe máy bị che khuất ở xa (Recall máy trên gold chỉ `45.5%`), chỉ dùng pre-label làm gợi ý vị trí, tuyệt đối không nghiệm thu nhanh.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Quy tắc Ẩn nhãn 15 giây đầu (Chống lọt `missing`):** Ở 100% ảnh, bắt buộc bấm phím `H` (ẩn toàn bộ box) trong 15 giây đầu tiên để quét theo 4 góc phần tư và đếm nhẩm số `person` ở sát mép ảnh trước khi bật nhãn máy lên (kiểm chứng: tỉ lệ bắt bẫy `missing` ở vòng duyệt lại phải đạt $\ge 2/3$ và tỉ lệ `added` trên ảnh đông tiệm cận mức thiếu hụt Recall của máy).
  2. **Quy tắc Lọc theo lớp trong CVAT (Chống lọt `wrong_class`):** Sau khi duyệt xong từng ảnh, bật bộ lọc *Filter* của CVAT theo từng lớp dễ nhầm (`label == "bicycle"` và `label == "car"`) để kiểm tra 100% các box thuộc 2 lớp này xem có `person` hoặc `three_wheeler` nào bị gán nhầm không (kiểm chứng: bắt 3/3 bẫy `wrong_class`, thời gian tăng thêm $\le 15\text{ giây/ảnh}$).
  3. **Quy tắc Phóng to 200% cho ảnh đông $\ge 30$ vật (Chống lọt `loose` và box `< 20px`):** Với mọi ảnh có từ 30 box trở lên (như `img_005`, `img_018`, `img_021`, `img_024`, `img_028`, `img_029`), bắt buộc zoom 200% và dùng phím `Tab` đi qua từng đối tượng để chỉnh biên box lệch và gộp ngay các cụm xe ở xa cao `< 20px` vào `ignore_region` (kiểm chứng: `validate_labels.py` báo 0 cảnh báo `< 20px` ngay từ lần export đầu tiên).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | `sam2.1_b.pt` (SAM 2.1 Base) | **3 click** (`2 dương`: `400,480`, `600,110`; `1 âm`: `720,300`) | **Không gồm người lái và khách phía sau** nhờ 1 click âm tại lưng tài xế (`720,300`), nhưng vẫn dính một mảng nhỏ ở ống quần/dép tài xế; kính chắn gió trong suốt phía trước được tính vào mask xe. | Mask (`317,895 px = 45.5%` ảnh, 1 polygon) tràn nhẹ sang vành bánh trước của chiếc `motorbike` đỗ sát mũi xe bên phải do ranh giới tối màu tiếp giáp nhau. | `submission/masks/tuktuk_img013.jpg` |
| `img_032.jpg` | `sam2.1_b.pt` (SAM 2.1 Base) | **3 click + 1 box** (`2 dương`: `415,350`, `415,460`; `1 âm`: `570,350`; `box`: `285,260,535,565`) | **Không gồm người đứng cạnh** (nhờ click âm `570,350` vào áo đen); đệm ghế gắn cố định trên khoang khách được tính trọn vẹn vào mask xích lô. | Mask (`54,119 px = 9.2%` ảnh, 1 polygon) bị tràn một góc sang vách mui của chiếc xích lô thứ hai đỗ sát phía sau bên trái (hai vật cùng màu chạm nhau); phần bánh đạp phía sau bị người áo đen che khuất nên tách rời khỏi khoang trước. | `submission/masks/cyclo_img032.jpg` |
| `img_002.jpg` | `sam2.1_b.pt` (SAM 2.1 Base) | **4 click + 1 box** (`2 dương`: `500,560`, `480,430`; `2 âm`: `615,490`, `730,560`; `box`: `355,410,695,660`) | **Không gồm người lái tuk-tuk** (click âm `615,490`) và **không gồm người lái xe máy tiền cảnh** (click âm `730,560`). | Mask (`47,170 px = 6.8%` ảnh, 2 polygon) ôm tốt mui và gầm xe xanh, nhưng qua khoảng hở lớn giữa khung xe (nhìn xuyên qua thấy đám đông phía sau), mask bị lem vào phần váy/chân của khách bộ hành đứng sau khung sắt. | `submission/masks/tuktuk_img002.jpg` |

- **Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?**
  - **Quy tắc giúp ích rõ rệt:**
    - *Mục 4.2.2 (Lỗ thủng $< 5\%$ diện tích vật không vẽ lỗ, lỗ lớn bỏ ra khỏi polygon):* Giúp quyết định giữ nguyên các nan hoa bánh xe xích lô (`img_032.jpg`) và khung sắt trang trí nhỏ của tuk-tuk (`img_013.jpg`, `img_002.jpg`) trong 1 polygon liền mạch, trong khi khoảng trống lớn ở khoang giữa tuk-tuk (`img_013.jpg`, `img_002.jpg`) phải được khoét bỏ bằng click âm.
    - *Mục 4.2.3 (Bóng đổ không thuộc vật, kính xe thuộc vật):* Giúp xác định vùng kính chắn gió phía trước của tuk-tuk ở `img_013.jpg` thuộc về mask xe, và phần bóng đen dưới gầm xe ở `img_002.jpg`, `img_013.jpg` không thuộc mask.
    - *Mục 4.2.4 (Hai vật chạm nhau không chồng polygon; vùng chạm thuộc vật phía trước):* Giải thích vì sao phải dùng click âm để tách người lái/người đứng che phía trước ra khỏi mask xe ở cả 3 ảnh.
  - **Quy tắc còn thiếu trong guideline v1.0 (cần bổ sung vào v1.1):**
    1. **Một vật bị người/xe phía trước cắt rời thành 2 mảnh không liên thông (Disjoint / Multi-polygon occlusion):** Ở `img_002.jpg` (mũi tuk-tuk bên phải bị người lái cắt tách khỏi thân xe, tạo ra `2 polygon`) và `img_032.jpg` (khung đạp phía sau xích lô bị người đứng che khuất hoàn toàn phần nối với khoang khách), guideline v1.0 chưa quy định rõ là gộp chung các mảnh rời vào 1 đối tượng multi-polygon, chỉ lấy mảnh lớn nhất, hay bỏ phần nhỏ bị che.
    2. **Kính chắn gió / khung mui trong suốt nhìn xuyên qua thấy người bên trong (Câu hỏi mở #5):** Nếu tính kính xe thuộc mask xe nhưng phía sau kính lại là khuôn mặt/thân của `person`, thì vùng pixel chồng lấp đó thuộc polygon `three_wheeler`/`car` hay thuộc polygon `person`?
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Thực hiện qua script mô phỏng tương tác click điểm `sam2_points.py` (tương đương cơ chế positive/negative point prompt + bounding box của AI Tools Interactor trên CVAT). So với chạy batch offline từ box của YOLO26 (`r04`), việc chủ động đặt 1–2 click âm (`--neg`) vào người lái và kết hợp `--box` giúp khắc phục triệt để lỗi chỉ click 1 điểm dương vào thân dưới chỉ lấy được bánh xe (như lần thử đầu trên `img_002.jpg` chỉ ra `0.7%` diện tích ảnh).

## 6. Calibration (Part 6)

- **Số quyết định:** **5 quyết định** trong [submission/qc/calibration_decisions.csv](file:///Users/hung/hacker/VinUni/K5-L2-Day08-PhamHuuHung-2A202603087-AI-Assisted-Labeling/submission/qc/calibration_decisions.csv) (`CAL-01` đến `CAL-05`, bao gồm 3 ca `cau_hoi_mo`, 1 ca `loi_may`, 1 ca `bay_lot`).
  - **Quyết định quan trọng nhất:**
    - **`CAL-02` (`loi_may` trên `img_002.jpg`, `img_032.jpg`):** Xử lý xung đột giữa detector đóng (YOLO26 gọi bánh sau xích lô là `bicycle 0.74` và gọi tuk-tuk là `car`) và detector mở (YOLOE sinh 2 box `three_wheeler` lồng nhau trên `img_002.jpg` hoặc chỉ ôm khoang khách trước của xích lô ở `img_032.jpg`). Chốt quy tắc: xóa box trùng/sai lớp, kéo 1 box `three_wheeler` duy nhất ôm trọn cả khoang khách, mui và phần khung xe đạp/xe máy gắn liền.
    - **`CAL-01` & `CAL-03` (`cau_hoi_mo` trên `img_005.jpg`, `img_013.jpg`, `img_018.jpg`, `img_002.jpg`):** Thay ngưỡng ước lượng cảm tính "còn thấy $\ge 20\%$" cho người ngồi trong xe có mui bằng tiêu chí cụ thể **"nhìn thấy rõ ít nhất đầu và vai"**, đồng thời quy định dùng click âm (`--neg`) để khoét trừ `person` khỏi mask xe (chấp nhận multi-polygon khi người đứng chắn cắt đôi thân xe như ở `img_002.jpg`).
- **Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:**
  1. Chốt toàn bộ 7 câu hỏi mở của guideline v1.0 (phân biệt xe đạp điện/xe máy điện qua bàn đạp; ngưỡng gộp `ignore_region` khi cụm $\ge 5$ xe cao `< 20px` hoặc che khuất $> 80\%$; hàng hóa nhô $\le 20\%$ chiều dài xe; tiêu chí "đầu và vai" cho `person` trong mui xe; khoét trừ `person` khỏi mask kính/khung xe và cho phép multi-polygon; không bao giờ auto-accept theo `conf`).
  2. Bổ sung bảng 4 nhóm ca khó mới kèm ảnh minh họa (`img_002`, `img_005`, `img_013`, `img_018`, `img_021`, `img_025`, `img_026`, `img_032`) và bảng lịch sử phiên bản `v1.1`.
- **Nguyên nhân gốc (phân biệt rõ 3 nguồn):**
  - **Do guideline v1.0 cố ý để mở / mơ hồ (`CAL-01`, `CAL-03`, `CAL-04`):** Chưa định nghĩa cách đo "20% diện tích bị che" bằng mắt, chưa nói rõ multi-polygon khi vẽ mask bị người cắt ngang, và chưa chốt ranh giới cụm xe dày đặc ở trung cảnh $\rightarrow$ **Cần sửa và bổ sung chữ vào `guideline_v1.1_calibrated.md`**.
  - **Do hạn chế kiến trúc của máy (`CAL-02`):** Tập nhãn cố định của COCO không có `three_wheeler`, còn text prompt của YOLOE nhạy với cả từng bộ phận xe $\rightarrow$ **Giải quyết bằng kết hợp model + quy tắc hậu xử lý/kéo box**.
  - **Do lỗi quy trình của người duyệt (`CAL-05`):** Lọt 5/12 bẫy (`img_005`, `img_018`, `img_021`, `img_025`, `img_026`) và để sót 12 box `< 20px` ở lần export đầu tiên không phải vì guideline sai, mà vì **thiên kiến tự động hóa** khi duyệt ảnh đông $\rightarrow$ **Giải quyết bằng siết quy trình checklist 4 bước và chạy lại vòng Rework/QC**, không làm méo định nghĩa lớp trong guideline.

## 7. QC và quyết định (Part 6)

- **Mẫu:** **5/24 ảnh batch** (`0` ảnh `can_review` + `5` ảnh ngẫu nhiên lấy theo `seed = 42`: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`), tổng cộng **102 đối tượng** (`img_003`: 20, `img_006`: 30, `img_014`: 19, `img_031`: 3, `img_035`: 30). Người kiểm: **Phạm Hữu Hưng** (**hạn chế:** người thực hiện QC cũng chính là người đã duyệt lô batch, nên dễ lặp lại cùng điểm mù nhận thức hoặc thiên kiến xác nhận trên những vật đã bỏ sót trước đó).
- **Lỗi theo đối tượng:** **4/102 = 3.9%** (mẫu số: `102` đối tượng trong 5 ảnh mẫu) · **Lỗi theo ảnh:** **3/5 = 60.0%** (mẫu số: `5` ảnh trong mẫu; ngưỡng Release yêu cầu $\le 2.0\%$).
- **Top lỗi + một ảnh ví dụ mỗi loại:**
  1. **`thua` (2/4 lỗi, chiếm 50%):**
     - `img_003.jpg` (đối tượng `#12 three_wheeler 0.32`): Box dư của YOLOE ở hậu cảnh bên trái ôm nhầm phần mái hiên và cụm xe máy ở xa nhưng chưa bị xóa khi review.
     - `img_006.jpg` (đối tượng `#83 car 0.59`): Box dương tính giả của YOLO26 ở cuối hẻm phố đi bộ Tạ Hiện (nhầm biển hiệu/mái che ở xa thành ô tô).
  2. **`bo_sot` (1/4 lỗi, chiếm 25%):**
     - `img_003.jpg` (không có ID): Bỏ sót 1 chiếc `motorbike` chở sọt hàng màu xanh đỗ ngay phía sau bên trái người đi bộ `#17 person 0.85`.
  3. **`sai_vi_tri_kich_thuoc` (1/4 lỗi, chiếm 25%):**
     - `img_031.jpg` (đối tượng `#433 bicycle 0.36`): Cạnh trái của box chỉ ôm đến bánh sau mà cắt mất nửa bên trái của mẹt hoa quả gắn trên yên sau xe đạp (theo `CAL-04`, giỏ/mẹt hàng gắn trên yên trong phạm vi $\le 20\%$ cần được ôm đủ).
- **Giới hạn mẫu nhỏ (cận trên 95% trong `qc_report_auto.md`):**
  - Với **3/5 ảnh có lỗi**, cận trên khoảng tin cậy 95% (Wilson) của tỉ lệ ảnh lỗi thực sự trong toàn lô lên tới **88%**.
  - Ngay cả trong trường hợp lý tưởng có **0/5 ảnh lỗi (`0%`)**, theo quy tắc thống kê $3/n$ (cận trên 95%), tỉ lệ ảnh lỗi thật vẫn có thể lên tới $3/5 = \mathbf{60\%}$ (hoặc ~43% theo Wilson). Do đó, với cỡ mẫu nhỏ ($n = 5$ ảnh trên tổng 24 ảnh), **không thể chứng minh về mặt thống kê rằng lô dữ liệu đạt ngưỡng lỗi $\le 2\%$**; mẫu nhỏ chỉ có giá trị phát hiện nhanh lỗi để bác bỏ (Rework), không đủ lực thống kê để khẳng định chất lượng dưới 2%.
- **Script kiểm được / không kiểm được:**
  - **Kiểm được (`validate_labels.py`, `review_stats.py`, `qc_report.py`, `check_submission.py`):** Đúng định dạng JSON COCO, đủ số lượng ảnh (`4` gold, `24` batch), tên lớp nằm trong taxonomy, tọa độ box không tràn khỏi khung ảnh, cảnh báo box cao `< 20px`, tỉ lệ giữ/sửa/xóa/thêm so với pre-label, tỉ lệ bắt 12 bẫy đã gài và tổng hợp số liệu từ phiếu `qc_review.csv`.
  - **Không kiểm được:** Box có thực sự ôm sát biên vật thể ($\le 2\text{ px}$) hay không (như `#433` ở `img_031.jpg`), một box có phải là ảo giác của máy trên nền đường/biển hiệu không (như `#83 car` ở `img_006.jpg` hay `#12 three_wheeler` ở `img_003.jpg`), các đối tượng mà **cả máy lẫn người đều cùng bỏ sót** trên 24 ảnh batch (vì không có ground-truth cho batch), và người duyệt có thực sự ẩn nhãn 15 giây đầu theo checklist hay không.

**Quyết định: Rework**, ngày **2026-11-17**.
- **Lý do (đi theo cây quyết định Hold $\rightarrow$ Rework $\rightarrow$ Release):**
  1. **Xét Hold:** Bộ ảnh không bị hỏng/mờ và cả 7 câu hỏi mở của guideline v1.0 đã được chốt rõ ràng thành quy tắc trong `guideline_v1.1_calibrated.md` (`--guideline-gap 0`, `--data-problem 0`) $\rightarrow$ **Không Hold**.
  2. **Xét Rework:** Lô nhãn vi phạm 2 tiêu chí chất lượng cốt lõi: (a) Tỉ lệ ảnh có lỗi trong mẫu QC là **3/5 = 60.0% > 2.0%** (tỉ lệ đối tượng lỗi **4/102 = 3.9%**); và (b) Tỉ lệ bắt bẫy chỉ đạt **7/12 = 58.3% < 80.0%** (lọt 5 bẫy ở `img_005`, `img_018`, `img_021`, `img_025`, `img_026`) $\rightarrow$ **Bắt buộc dừng ở Rework** (không được phép Release).
- **Nếu Rework / Hold, việc cần làm tiếp:**
  1. Mở lại Task `batch-review` trên CVAT, sửa ngay 4 lỗi đã phát hiện trong mẫu QC (`img_003.jpg`: xóa `#12`, vẽ thêm `motorbike` sau `#17`; `img_006.jpg`: xóa `#83 car`; `img_031.jpg`: kéo rộng `#433 bicycle`) và sửa 5 ảnh có bẫy lọt (`img_005`, `img_018`, `img_021`, `img_025`, `img_026`).
  2. Áp dụng 3 quy tắc chống thiên kiến (ẩn nhãn 15s đầu quét mép ảnh; dùng Filter kiểm tra 100% lớp `bicycle`, `car`, `three_wheeler`; zoom 200% trên 6 ảnh đông `005, 018, 021, 024, 028, 029`).
  3. Xuất các ảnh đã sửa ra `submission/labels/qc_fixed.json`, chạy lại `python scripts/build_release.py` để cập nhật `submission/release/final_det.coco.json`, rồi lấy mẫu QC vòng 2 (`qc_sample.py --force`) để nghiệm thu lại.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Dùng trợ lý AI (Antigravity) để hướng dẫn lệnh CLI, giải thích lỗi môi trường (`mobileclip2_b.ts` timeout) và rà soát quy trình; toàn bộ nhãn Gold (`gold_manual.json`) tự vẽ tay 100% trên CVAT và kiểm chứng qua `validate_labels.py`.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? **Không** — gán tay hoàn toàn 4 ảnh gold (`img_001.jpg`, `img_012.jpg`, `img_020.jpg`, `img_036.jpg`) trên CVAT trước khi chạy bất kỳ lệnh `prelabel.py` nào.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: macOS (Apple Silicon, GPU Apple MPS, Python 3.10.0, PyTorch 2.14.1, Ultralytics 8.4.123), CVAT chạy qua Docker local tại `http://localhost:8080`. Khi gán Gold lần đầu bị cảnh báo 24 box cao `< 20px` (98 đối tượng $\rightarrow$ sửa lần 1 còn 86 $\rightarrow$ sửa lần 2 chuẩn 73 đối tượng: `person`: 34, `motorbike`: 22, `car`: 8, `three_wheeler`: 7, `bicycle`: 2).
- Dùng cache `data/cache/` thay vì tự chạy? **Không** — đã tự chạy trực tiếp toàn bộ 6 lượt model (`r01`–`r06`) trên GPU Apple MPS.
- Lỗi / bug script đã gặp (nếu có): Khi chạy `yoloe` lần đầu, Ultralytics tự cài `clip` thành công nhưng khi tải `mobileclip2_b.ts` thì hàm `attempt_download_asset` gọi `api.github.com` bị `ConnectTimeoutError` dẫn đến `ValueError: The provided filename mobileclip2_b.ts does not exist`. Cách khắc phục: tải trực tiếp `mobileclip2_b.ts` từ GitHub Releases về `models/mobileclip2_b.ts`.
- Một điều bạn làm khác lần sau: Ngay khi vẽ tay trên CVAT, kiểm tra kỹ kích thước chiều cao box ($\ge 20\text{ px}$) và dùng `ignore_region` cho cụm vật nhỏ ở xa ngay từ lượt đầu để không phải export và sửa lại 3 lần.
