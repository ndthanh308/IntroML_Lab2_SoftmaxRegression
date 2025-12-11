# MNIST Digit Recognition with Softmax Regression

🔢 **Nhận dạng chữ số viết tay MNIST sử dụng Softmax Regression và Feature Engineering**

---

## 📖 Mô tả Bài toán

Dự án này thực hiện **nhận dạng chữ số viết tay** (0-9) từ bộ dữ liệu MNIST bằng mô hình **Softmax Regression** được lập trình từ đầu (from scratch) chỉ sử dụng **NumPy**. 

### Mục tiêu chính:
1. **Xây dựng Softmax Regression từ đầu** (không dùng thư viện học máy như scikit-learn, TensorFlow, PyTorch)
2. **Nghiên cứu các phương pháp Feature Engineering** để cải thiện độ chính xác
3. **So sánh hiệu suất** giữa 5 phương pháp biến đổi đặc trưng khác nhau
4. **Triển khai ứng dụng web** cho người dùng cuối sử dụng Streamlit

### Kết quả đạt được:
- ✅ **Độ chính xác cao nhất: 92.49%** (PCA 256 components)
- ✅ **5 phương pháp Feature Engineering** được thực nghiệm và đánh giá
- ✅ **Ứng dụng web tương tác** cho phép vẽ chữ số hoặc tải ảnh từ file
- ✅ **Tài liệu chi tiết** với phân tích toán học và trực quan hóa

---

## 👥 Thông tin Nhóm

**Group 09**

| STT | Họ và Tên | MSSV | Công việc chính |
|-----|-----------|------|-----------------|
| 1   | **Bùi Huy Giáp** | 23127289 | Dẫn xuất công thức Softmax & Cross-entropy; Cài đặt mô hình Softmax Regression; Feature Engineering: PCA & Rotation Invariance; Tổng hợp báo cáo cuối |
| 2   | **Lê Minh Đức** | 23127351 | Feature Engineering: Sobel Edge Detection; Viết các hàm metrics (Accuracy, Precision, Recall, F1-score); Viết báo cáo |
| 3   | **Vũ Tiến Dũng** | 23127354 | Feature Engineering: Baseline; Tổng hợp Metrics và so sánh các mô hình; Viết báo cáo |
| 4   | **Đinh Xuân Khương** | 23127398 | Feature Engineering: Average Pooling; Xây dựng ứng dụng Streamlit nhận dạng chữ số; Viết báo cáo |
| 5   | **Nguyễn Đồng Thanh** | 23127538 | Dẫn xuất công thức Gradient Descent; Tạo framework trình bày Feature Vector; Tiền xử lý dữ liệu MNIST; Chuẩn bị submission files; Quay video demo; Tổng hợp báo cáo cuối |

**Môn học**: Nhập môn Học máy (Introduction to Machine Learning)  
**Giảng viên**: Thầy Bùi Tiến Lên, Thầy Lê Nhựt Nam, Thầy Võ Nhật Tân
**Học kỳ**: HK1 2025-2026

---

## 📂 Cấu trúc Thư mục

```
IntroML_Lab2_SoftmaxRegression/
│
├── 📄 App_demo.py                    # Ứng dụng Streamlit triển khai mô hình
├── 📄 requirements.txt               # Danh sách thư viện cần thiết
├── 📄 README.md                      # Tài liệu này
├── 📄 LICENSE                        # Giấy phép mã nguồn
│
├── 📁 data/
│   └── 📁 raw/
│       └── 📄 mnist.npz              # Dữ liệu MNIST (tự động tải về khi chạy lần đầu)
│
├── 📁 lib/
│   ├── 📄 SoftmaxRegression.py       # Class mô hình Softmax Regression
│   └── 📄 helpers.py                 # Các hàm tiện ích (load data, metrics, visualization)
│
├── 📁 models/
│   └── 📄 best_model_weights.npz     # Trọng số mô hình tốt nhất (PCA 256, accuracy: 92.78%)
│
└── 📁 notebooks/
    ├── 📓 1_Implementation.ipynb           # Notebook 1: Cài đặt Softmax Regression cơ bản
    └── 📓 2_Feature_Experiments.ipynb      # Notebook 2: Thực nghiệm 5 phương pháp Feature Engineering
```

### Chi tiết các file quan trọng:

#### 1. **`lib/SoftmaxRegression.py`**
Chứa class `SoftmaxRegression` với các phương thức:
- `__init__()`: Khởi tạo trọng số W và bias b
- `softmax()`: Hàm kích hoạt Softmax có ổn định số học
- `forward()`: Lan truyền xuôi (Z = XW + b)
- `compute_loss()`: Tính Cross-Entropy Loss
- `backward()`: Tính gradient (đạo hàm) của W và b
- `fit()`: Huấn luyện mô hình bằng Mini-batch Gradient Descent
- `predict()`: Dự đoán nhãn (argmax)
- `predict_proba()`: Dự đoán xác suất các lớp
- `save_weights()` / `load_weights()`: Lưu/đọc trọng số mô hình

#### 2. **`lib/helpers.py`**
Các hàm hỗ trợ:
- `load_mnist_data()`: Tải và tiền xử lý MNIST (tự động tải nếu chưa có)
- `one_hot_encode()`: Mã hóa one-hot cho nhãn
- `compute_confusion_matrix()`: Tính ma trận nhầm lẫn
- `compute_metrics()`: Tính Accuracy, Precision, Recall, F1-Score
- `plot_confusion_matrix()`: Vẽ heatmap ma trận nhầm lẫn
- `plot_loss_curve()`: Vẽ đồ thị Loss qua các epoch

#### 3. **`notebooks/1_Implementation.ipynb`**
**Nội dung**:
- Lý thuyết Softmax Regression (công thức toán học)
- Cài đặt từ đầu (Forward, Backward, Gradient Descent)
- Huấn luyện trên MNIST baseline (784 features)
- Đánh giá: Accuracy, Confusion Matrix, Loss Curve
- **Kết quả**: 92.07% accuracy, 12.88s training time

#### 4. **`notebooks/2_Feature_Experiments.ipynb`**
**Nội dung**: Thực nghiệm 5 phương pháp biến đổi đặc trưng:

| Phương pháp | Số features | Accuracy | Training Time | Đặc điểm |
|-------------|-------------|----------|---------------|----------|
| **Baseline** | 784 | 92.07% | 12.88s | Flatten ảnh 28×28 trực tiếp |
| **Pooling** (pool=2) | 196 | 92.59% | 20.45s | Average pooling 2×2, giữ cấu trúc không gian |
| **Sobel** | 784 | 90.06% | 36.82s | Phát hiện biên (edge detection) |
| **PCA** (n=256) | 256 | **92.78%** | 26.62s | **Giảm chiều dữ liệu, 97.48% variance** ✨ |
| **Rotation** | 784 | 81.05% | 54.21s | Căn chỉnh theo trục chính |

**Kết luận**: **PCA 256 components** đạt độ chính xác cao nhất (92.78%) và được lưu vào `models/best_model_weights.npz`.

#### 5. **`App_demo.py`**
Ứng dụng Streamlit với 3 chức năng:
- **Tab 1**: Vẽ chữ số bằng chuột trên canvas
- **Tab 2**: Tải ảnh chữ số từ file (JPG, PNG)
- **Tab 3**: Demo với 10,000 ảnh từ MNIST test set

**Tính năng**:
- Hiển thị xác suất dự đoán cho 10 chữ số (0-9)
- Biểu đồ thanh (bar chart) cho top-3 dự đoán
- Hiển thị ảnh sau tiền xử lý (28×28 grayscale)
- Sử dụng mô hình PCA 256 đã huấn luyện

#### 6. **`models/best_model_weights.npz`**
File chứa trọng số mô hình tốt nhất:
- `pca_mean`: Vector trung bình của dữ liệu training (để chuẩn hóa)
- `pca_vt`: Ma trận chiếu PCA (256×784)
- `pca_n_components`: 256
- `pca_explained_variance`: Tỷ lệ phương sai giữ lại (97.48%)
- `model_weights`: W (256×10)
- `model_bias`: b (1×10)
- `scaler_min`, `scaler_max`: Giá trị min/max cho chuẩn hóa [0, 1]

---

## 🔧 requirements.txt - Chi tiết

File `requirements.txt` liệt kê **9 thư viện Python** cần thiết để chạy dự án:

```txt
numpy                      # Thư viện tính toán ma trận và vector (CORE)
pandas                     # Xử lý và phân tích dữ liệu dạng bảng
matplotlib                 # Vẽ đồ thị, biểu đồ (Loss curve, confusion matrix)
opencv-python              # Xử lý ảnh (Sobel edge detection)
seaborn                    # Vẽ heatmap đẹp cho confusion matrix
ipywidgets                 # Widget tương tác trong Jupyter Notebook
streamlit                  # Framework tạo ứng dụng web
streamlit-drawable-canvas  # Component vẽ trên canvas trong Streamlit
pillow                     # Xử lý và chuyển đổi ảnh (PIL)
```

### Giải thích từng thư viện:

| Thư viện | Vai trò | Sử dụng trong dự án |
|----------|---------|---------------------|
| **numpy** | Tính toán ma trận, vector, đại số tuyến tính | Xây dựng Softmax Regression (forward, backward, gradient descent), PCA transformation |
| **pandas** | Xử lý dữ liệu dạng bảng (DataFrame) | Tổ chức kết quả thực nghiệm, bảng so sánh metrics |
| **matplotlib** | Vẽ đồ thị 2D (line plot, bar chart, heatmap) | Vẽ Loss curve, confusion matrix, biểu đồ so sánh |
| **opencv-python** | Xử lý ảnh (filter, edge detection, morphology) | Áp dụng Sobel filter để phát hiện biên trong Feature Engineering |
| **seaborn** | Vẽ biểu đồ thống kê đẹp (built on matplotlib) | Vẽ heatmap confusion matrix với màu sắc chuyên nghiệp |
| **ipywidgets** | Tạo widget tương tác trong Jupyter (slider, button) | Tạo giao diện tương tác để chọn hyperparameters trong notebook |
| **streamlit** | Framework tạo ứng dụng web nhanh chóng | Xây dựng App_demo.py với giao diện đẹp và tương tác |
| **streamlit-drawable-canvas** | Component vẽ canvas trong Streamlit | Cho phép người dùng vẽ chữ số bằng chuột trên web |
| **pillow** | Xử lý ảnh (load, resize, convert format) | Đọc ảnh từ file upload, chuyển đổi sang numpy array |

---

## 🚀 Hướng dẫn Cài đặt và Chạy

### Bước 1: Clone Repository

```bash
git clone https://github.com/[username]/IntroML_Lab2_SoftmaxRegression.git
cd IntroML_Lab2_SoftmaxRegression
```

### Bước 2: Tạo Virtual Environment (Khuyến nghị)

**Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/MacOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt Thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý**: Nếu gặp lỗi với `opencv-python`, thử cài đặt:
> ```bash
> pip install opencv-python-headless
> ```

### Bước 4: Chạy Jupyter Notebooks

#### Option 1: Jupyter Notebook (Classic)
```bash
jupyter notebook
```
Sau đó mở trình duyệt và truy cập:
- `notebooks/1_Implementation.ipynb` - Cài đặt cơ bản
- `notebooks/2_Feature_Experiments.ipynb` - Thực nghiệm Feature Engineering

#### Option 2: Jupyter Lab (Modern)
```bash
jupyter lab
```

#### Option 3: VS Code
- Mở file `.ipynb` trong VS Code
- Chọn kernel Python (từ venv đã tạo)
- Chạy từng cell bằng Shift+Enter

### Bước 5: Chạy Ứng dụng Streamlit

```bash
streamlit run App_demo.py
```

Hoặc chỉ định port cụ thể:
```bash
streamlit run App_demo.py --server.port 8501
```

Ứng dụng sẽ mở tại: **http://localhost:8501**

### Bước 6: Sử dụng Ứng dụng

1. **Tab "Vẽ Chữ Số"**: 
   - Vẽ chữ số bằng chuột trên canvas trắng
   - Nhấn nút "Dự đoán" để nhận kết quả
   - Xem xác suất dự đoán cho 10 chữ số

2. **Tab "Tải Ảnh"**:
   - Upload ảnh chữ số (JPG, PNG, etc.)
   - Hệ thống tự động tiền xử lý (resize, grayscale, normalize)
   - Hiển thị dự đoán với độ tin cậy

3. **Tab "MNIST Demo"**:
   - Xem 10,000 ảnh từ MNIST test set
   - Điều chỉnh slider để chọn ảnh
   - Xem dự đoán và nhãn thực tế

---

## 📊 Kết quả Thực nghiệm

### So sánh 5 Phương pháp Feature Engineering:

| Phương pháp | Features | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Training Time (s) | Efficiency Ratio* |
|-------------|----------|--------------|---------------|------------|--------------|-------------------|-------------------|
| **Baseline** | 784 | 92.07 | 92.17 | 92.07 | 92.06 | 12.88 | 0.0715 |
| **Pooling (2×2)** | 196 | 92.59 | 92.68 | 92.59 | 92.59 | 20.45 | **0.0453** |
| **Sobel Edge** | 784 | 90.06 | 90.23 | 90.06 | 90.03 | 36.82 | 0.0245 |
| **PCA (256)** | 256 | **92.78** ⭐ | **92.86** | **92.78** | **92.78** | 26.62 | 0.0349 |
| **Rotation** | 784 | 81.05 | 81.53 | 81.05 | 80.85 | 54.21 | 0.0150 |

*Efficiency Ratio = Accuracy / (Features × Training Time)

### Nhận xét:
- ✅ **PCA 256** đạt độ chính xác cao nhất: **92.78%** (cao hơn baseline 0.71%)
- ✅ **Pooling 2×2** có efficiency ratio tốt nhất (giảm 75% features, vẫn giữ 92.59% accuracy)
- ✅ **Sobel** và **Rotation** không hiệu quả (độ chính xác thấp hơn, thời gian huấn luyện lâu hơn)
- ✅ **PCA** cân bằng tốt giữa độ chính xác, số features, và thời gian huấn luyện

### Mô hình tốt nhất được chọn:
🏆 **PCA 256 components** (saved in `models/best_model_weights.npz`)
- Accuracy: **92.78%**
- Explained Variance: **97.48%**
- Training Time: **26.62s**
- Features: **256** (giảm 67.3% so với baseline 784)

---

## 📚 Tài liệu Tham khảo

### Bộ dữ liệu:
- **MNIST Database**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- Kích thước: 60,000 training images + 10,000 test images
- Format: 28×28 grayscale images (0-255)
- 10 classes: digits 0-9

### Thư viện sử dụng:
- [NumPy](https://numpy.org/) - Tính toán khoa học
- [Matplotlib](https://matplotlib.org/) - Trực quan hóa
- [Streamlit](https://streamlit.io/) - Ứng dụng web
- [OpenCV](https://opencv.org/) - Xử lý ảnh
- [Seaborn](https://seaborn.pydata.org/) - Vẽ biểu đồ thống kê

### Tài liệu học thuật:
- [Softmax Regression - CS229 Stanford](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
- [PCA for Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [MNIST Handwritten Digit Classification](https://en.wikipedia.org/wiki/MNIST_database)

---

## ⚠️ Lưu ý và Xử lý Lỗi

### Lỗi thường gặp:

#### 1. **ModuleNotFoundError: No module named 'streamlit'**
**Nguyên nhân**: Chưa cài đặt thư viện hoặc chạy sai Python environment  
**Giải pháp**:
```bash
pip install -r requirements.txt
```
Hoặc activate virtual environment trước:
```bash
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac
```

#### 2. **FileNotFoundError: mnist.npz not found**
**Nguyên nhân**: Dữ liệu MNIST chưa được tải về  
**Giải pháp**: Chạy notebook lần đầu tiên, hàm `load_mnist_data()` sẽ tự động tải về vào `data/raw/`

#### 3. **Streamlit error: "Please run it as: streamlit run App_demo.py"**
**Nguyên nhân**: Chạy `python App_demo.py` thay vì `streamlit run`  
**Giải pháp**:
```bash
streamlit run App_demo.py
```

#### 4. **Lỗi "port already in use"**
**Nguyên nhân**: Port 8501 đang được sử dụng  
**Giải pháp**: Chỉ định port khác:
```bash
streamlit run App_demo.py --server.port 8502
```

#### 5. **Notebook kernel crash khi chạy PCA**
**Nguyên nhân**: Thiếu RAM (PCA với 60,000×784 matrix tốn ~300MB)  
**Giải pháp**: Giảm số samples hoặc tăng RAM, hoặc chạy trên Google Colab

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Nếu bạn muốn cải thiện dự án:
1. Fork repository
2. Tạo branch mới: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Tạo Pull Request

---

## 📧 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng liên hệ qua:
- Email: [email nhóm]
- GitHub Issues: [Link to issues page]

---

**Cảm ơn bạn đã xem dự án! 🎉**
