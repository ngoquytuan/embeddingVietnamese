Từ báo cáo, tôi thấy có 6 models trong config nhưng chỉ 5 models được đánh giá thành công. Model `dangvantuan/vietnamese-embedding` có thể đã gặp lỗi trong quá trình load hoặc đánh giá.

## Phân tích kết quả từ báo cáo:

### 🏆 **Ranking theo hiệu suất với tiếng Việt:**

**1. AITeamVN/Vietnamese_Embedding** - ⭐ **KHUYẾN NGHỊ**
- **Avg Similarity**: 0.4529 (vừa phải - tốt)
- **Std Similarity**: 0.0921 (phân tán thấp - ổn định)
- **Similarity Range**: 0.5816 (phạm vi rộng - phân biệt tốt)
- **High Sim Ratio**: 0.11% (rất thấp - không over-generalize)
- **Embedding Dim**: 1024 (cao nhất - giàu thông tin)
- ✅ **Tối ưu cho tiếng Việt**, cân bằng tốt giữa similarity và discrimination

**2. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**
- **Avg Similarity**: 0.4974 (tương tự AITeamVN)
- **Std Similarity**: 0.1733 (phân tán cao - phân biệt tốt)
- **Similarity Range**: 0.8950 (rộng nhất - phân biệt xuất sắc)
- **High Sim Ratio**: 9.20% (thấp - tốt)
- ✅ **Lựa chọn tốt thứ 2**, đặc biệt tốt cho phân biệt ngữ nghĩa

**3. sentence-transformers/distiluse-base-multilingual-cased**
- **Avg Similarity**: 0.4136 (thấp nhất - conservative)
- **Similarity Range**: 0.8241 (rộng - phân biệt tốt)
- **High Sim Ratio**: 1.06% (rất thấp)
- ✅ **Tốt cho retrieval** khi cần độ chính xác cao

### ❌ **Models không phù hợp:**

**4. sentence-transformers/all-MiniLM-L6-v2**
- **Avg Similarity**: 0.7538 (quá cao)
- **High Sim Ratio**: 80.23% (quá cao - over-generalize)
- ❌ Không phù hợp với tiếng Việt, coi mọi thứ đều giống nhau

**5. cross-encoder/ms-marco-MiniLM-L-6-v2** 
- **Avg Similarity**: 0.9821 (cực cao - không phân biệt được)
- **High Sim Ratio**: 100% (tất cả đều > threshold)
- ❌ **Worst choice** - model này là cross-encoder không phải bi-encoder, không phù hợp cho embedding

### 🔍 **Model bị missing:**
`dangvantuan/vietnamese-embedding` không xuất hiện trong kết quả → có thể:
- Model không tồn tại trên HuggingFace
- Lỗi download/load
- Kiểm tra log file để xem lỗi cụ thể

## 📊 **Khuyến nghị sử dụng:**

### **Cho Production (tiếng Việt):**
1. **AITeamVN/Vietnamese_Embedding** - Tối ưu nhất
2. **paraphrase-multilingual-MiniLM-L12-v2** - Backup tốt

### **Cho nghiên cứu/thử nghiệm:**
- **distiluse-base-multilingual-cased** - Conservative, ít false positive

### **Tránh sử dụng:**
- `all-MiniLM-L6-v2` và `cross-encoder/ms-marco-MiniLM-L-6-v2`

## 🛠 **Để khắc phục model missing:**

Thêm vào code để debug model nào bị lỗi:

```python
# Thêm vào phần evaluate_all_models:
for i, model_config in enumerate(models, 1):
    model_name = model_config["name"]
    logger.info(f"Đánh giá model {i}/{len(models)}: {model_name}")
    
    try:
        result = self.evaluate_model(model_name, chunks)
        # ... existing code
    except Exception as e:
        logger.error(f"❌ Model {model_name} failed: {e}")
        results[model_name] = {
            "model_name": model_name,
            "status": "error", 
            "error": str(e)
        }
```

Bạn có muốn tôi cập nhật code để hiển thị cả models bị lỗi trong báo cáo không?