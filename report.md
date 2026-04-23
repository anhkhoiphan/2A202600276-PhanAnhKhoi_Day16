# Lab 16 — Cloud AI Environment Setup (GCP)
## Phương án CPU + LightGBM (Phần 7)

**Sinh viên:** Phan Anh Khôi  
**Student ID:** 2A202600276
**Project ID:** gen-lang-client-0008892973  
**Region/Zone:** asia-southeast1 / asia-southeast1-b  
**Instance:** n2-standard-8 (8 vCPU, 32 GB RAM)

---

## 1. Lý do sử dụng CPU thay GPU

Tài khoản GCP mới (Free Tier / $300 credit) mặc định có quota GPU NVIDIA T4 bằng 0. Yêu cầu tăng quota đã được gửi nhưng bị từ chối do tài khoản chưa có lịch sử thanh toán đủ lâu. Toàn bộ các zone thuộc region `us-central1` và `asia-southeast1` đều trả về lỗi *"not enough resources available"* khi cố gắng tạo VM gắn GPU. Do đó, bài lab chuyển sang phương án CPU với instance `n2-standard-8` và mô hình LightGBM (gradient boosting) trên tập dữ liệu Credit Card Fraud Detection (Kaggle), theo đúng hướng dẫn của Phần 7.

---

## 2. Kết quả chạy Benchmark

### 2.1 Screenshot terminal — output của `python3 benchmark.py`

> Link snapshot: https://prnt.sc/ySyt4WaxhM6T
> Nội dung cần thấy: toàn bộ output của benchmark bao gồm load time, training time và các metrics.

Kết quả đầy đủ đã được lưu vào `benchmark_result.json`:

```json
{
  "load_time_s": 1.222,
  "train_time_s": 0.827,
  "best_iteration": 1,
  "auc_roc": 0.94151,
  "accuracy": 0.998999,
  "f1_score": 0.742081,
  "precision": 0.666667,
  "recall": 0.836735,
  "inference_latency_1row_ms": 0.704,
  "inference_throughput_1000rows_ms": 0.9
}
```

### 2.2 Bảng tổng hợp kết quả

| Metric | Kết quả |
|--------|---------|
| Thời gian load data (284,807 rows) | 1.22 s |
| Thời gian training | 0.83 s |
| Best iteration | 1 |
| AUC-ROC | **0.9415** |
| Accuracy | 0.9990 |
| F1-Score | 0.7421 |
| Precision | 0.6667 |
| Recall | 0.8367 |
| Inference latency (1 row) | **0.70 ms** |
| Inference throughput (1000 rows) | **0.9 ms** |

---

## 3. Screenshot GCP Billing Reports

> Link: https://prnt.sc/dMcy7Vpx7Typ
> Truy cập: GCP Console → Billing → Reports → chọn khoảng thời gian hôm nay.  
> Cần thấy: các dịch vụ Compute Engine (n2-standard-8), Cloud NAT, Cloud Load Balancing và chi phí tương ứng.

---

## 4. So sánh CPU vs GPU

| Tiêu chí | GPU (n1-standard-4 + T4) | CPU (n2-standard-8) |
|----------|--------------------------|----------------------|
| Chi phí/giờ | ~$0.54 (GPU $0.35 + VM $0.19) | ~$0.43 |
| Yêu cầu quota | Cần xin riêng, thường bị từ chối | Không cần |
| Training LightGBM | Không hỗ trợ tăng tốc GPU | 0.83 s (đủ nhanh) |
| Inference 1 row | — | 0.70 ms |
| Phù hợp workload | LLM / deep learning | Tabular ML (LGBM, XGBoost) |

**Nhận xét:** Với bài toán tabular ML (LightGBM), `n2-standard-8` CPU thực ra **phù hợp hơn và rẻ hơn** GPU. LightGBM không tận dụng được GPU trong phần lớn các tác vụ thông thường. Phương án CPU là lựa chọn hợp lý về mặt kinh tế và kỹ thuật khi không có quota GPU.

---

## 5. Mã nguồn Terraform (đã chỉnh sửa)

Các thay đổi chính so với cấu hình GPU gốc:

- `machine_type`: `n1-standard-4` → `n2-standard-8`
- `gpu_count`: `1` → `0` (comment out toàn bộ `guest_accelerator` block)
- `scheduling.on_host_maintenance`: `TERMINATE` → `MIGRATE`
- `image`: DLVM CUDA image → `ubuntu-os-cloud/ubuntu-2204-lts`
- `user_data.sh`: bỏ toàn bộ NVIDIA/Docker setup, chỉ cài Python

Mã nguồn đầy đủ nằm trong thư mục `terraform-gcp/`.
