# Thông tin học viên

- Họ và tên: Phạm Hữu Hưng
- MSHV: 2A202603087
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: hungtun
- Repo bài nộp (URL): https://github.com/hungtun/K5-L2-Day08-PhamHuuHung-2A202603087-AI-Assisted-Labeling
- Commit chốt bài (hash): HEAD (xem git log của repo)
- Máy: macOS (Apple Silicon, GPU Apple MPS)
- CVAT (Docker): v2.81.0 (chạy local tại http://localhost:8080)

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 455 (86.7%) / 12 (2.3%) / 58 (11.0%) (mẫu số: 525 box máy); added: 27 (5.5%) (mẫu số: 494 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 62.5% (45/72 box máy) / 61.6% (45/73 đối tượng gold)
- Bắt bẫy: 7/12 = 58.3%
- Thời gian: gán tay 9.25 phút/ảnh (37 phút / 4 ảnh, 73 đối tượng); duyệt có pre-label 1.46 phút/ảnh (35 phút / 24 ảnh, 505 đối tượng)
- Lỗi QC: 60.0% theo ảnh (3/5 ảnh mẫu), 3.9% theo đối tượng (4/102 đối tượng mẫu)
- **Quyết định:** Rework — lý do một dòng: Tỉ lệ ảnh có lỗi trong mẫu QC là 3/5 = 60.0% (> 2%) và tỉ lệ bắt bẫy đạt 7/12 = 58.3% (< 80%), cần duyệt sửa lại theo guideline v1.1 và xuất qc_fixed.json.
