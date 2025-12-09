import numpy as np
import os
import urllib.request
import matplotlib.pyplot as plt

# --- 1. DATA LOADING & PROCESSING ---
def load_mnist_data(data_path='data/raw/mnist.npz'):
    """
    Tải và load dữ liệu MNIST, trả về vector phẳng (flatten).
    
    Args:
        data_path (str): Đường dẫn đến file mnist.npz. Mặc định: 'data/raw/mnist.npz'
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test)
            - x_train (np.ndarray): Dữ liệu training, shape (60000, 784), giá trị [0, 1]
            - y_train (np.ndarray): Nhãn training, shape (60000,), giá trị [0-9]
            - x_test (np.ndarray): Dữ liệu testing, shape (10000, 784), giá trị [0, 1]
            - y_test (np.ndarray): Nhãn testing, shape (10000,), giá trị [0-9]
    """
    # Tạo thư mục nếu chưa có
    directory = os.path.dirname(data_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Tải file nếu chưa có
    if not os.path.exists(data_path):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"Downloading MNIST from {url}...")
        urllib.request.urlretrieve(url, data_path)
    
    # Load data
    with np.load(data_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    # Chuẩn hóa về [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Flatten (N, 28, 28) -> (N, 784)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    return x_train, y_train, x_test, y_test

def one_hot_encode(y, num_classes=10):
    """
    Chuyển đổi nhãn số nguyên sang dạng one-hot encoding.
    
    Args:
        y (np.ndarray): Mảng nhãn dạng số nguyên, shape (m,), giá trị [0, num_classes-1]
        num_classes (int): Số lượng classes. Mặc định: 10
    
    Returns:
        np.ndarray: Ma trận one-hot, shape (m, num_classes)
                    Mỗi hàng có giá trị 1 tại vị trí class tương ứng, còn lại là 0
    
    Example:
        >>> y = np.array([0, 2, 1])
        >>> one_hot_encode(y, num_classes=3)
        array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.]])
    """
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    for i in range(m):
        one_hot[i, y[i]] = 1
    return one_hot

# --- 2. EVALUATION METRICS (Manual Implementation) ---
def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Tính Confusion Matrix bằng Numpy.
    
    Args:
        y_true (np.ndarray): Nhãn thực tế, shape (m,), giá trị [0, num_classes-1]
        y_pred (np.ndarray): Nhãn dự đoán, shape (m,), giá trị [0, num_classes-1]
        num_classes (int): Số lượng classes. Mặc định: 10
    
    Returns:
        np.ndarray: Confusion matrix, shape (num_classes, num_classes)
                    - Hàng i: Tất cả mẫu có nhãn thực tế là class i
                    - Cột j: Tất cả mẫu được dự đoán là class j
                    - cm[i, j]: Số lượng mẫu thuộc class i nhưng được dự đoán là class j
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm

def compute_metrics(y_true, y_pred, num_classes=10):
    """
    Tính các chỉ số đánh giá: Accuracy, Precision, Recall, F1-score (Macro-average).
    
    Args:
        y_true (np.ndarray): Nhãn thực tế, shape (m,), giá trị [0, num_classes-1]
        y_pred (np.ndarray): Nhãn dự đoán, shape (m,), giá trị [0, num_classes-1]
        num_classes (int): Số lượng classes. Mặc định: 10
    
    Returns:
        dict: Dictionary chứa các metrics:
            - "accuracy" (float): Tỷ lệ dự đoán đúng tổng thể
            - "precision" (float): Macro-averaged precision (trung bình của precision các class)
            - "recall" (float): Macro-averaged recall (trung bình của recall các class)
            - "f1_score" (float): Macro-averaged F1-score (trung bình của F1 các class)
            - "confusion_matrix" (np.ndarray): Confusion matrix, shape (num_classes, num_classes)
    
    Notes:
        - TP (True Positive): Số mẫu thuộc class i và được dự đoán đúng là class i
        - FP (False Positive): Số mẫu không thuộc class i nhưng bị dự đoán nhầm là class i
        - FN (False Negative): Số mẫu thuộc class i nhưng bị dự đoán nhầm là class khác
        - Precision = TP / (TP + FP): Độ chính xác của dự đoán
        - Recall = TP / (TP + FN): Khả năng phát hiện đúng
        - F1 = 2 * (Precision * Recall) / (Precision + Recall): Trung bình điều hòa
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
    # Accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(num_classes):
        tp = cm[i, i]  # True Positive: Phần tử trên đường chéo
        fp = np.sum(cm[:, i]) - tp  # False Positive: Tổng cột i trừ TP (các mẫu không phải class i nhưng dự đoán là i)
        fn = np.sum(cm[i, :]) - tp  # False Negative: Tổng hàng i trừ TP (các mẫu là class i nhưng dự đoán sai)
        
        # Precision = TP / (TP + FP)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall = TP / (TP + FN)
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 = 2 * (P * R) / (P + R)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        
    return {
        "accuracy": accuracy,
        "precision": np.mean(precisions),  # Macro average
        "recall": np.mean(recalls),        # Macro average
        "f1_score": np.mean(f1_scores),    # Macro average
        "confusion_matrix": cm
    }

# --- 3. VISUALIZATION ---
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Vẽ Confusion Matrix sử dụng Matplotlib với màu sắc và số liệu.
    
    Args:
        cm (np.ndarray): Confusion matrix cần vẽ, shape (n_classes, n_classes)
        classes (list or np.ndarray): Danh sách tên các classes, length = n_classes
        title (str): Tiêu đề của biểu đồ. Mặc định: 'Confusion Matrix'
    
    Returns:
        None: Hiển thị biểu đồ trực tiếp
    
    Notes:
        - Màu xanh đậm thể hiện giá trị cao (nhiều mẫu)
        - Màu xanh nhạt thể hiện giá trị thấp (ít mẫu)
        - Số liệu hiển thị màu trắng nếu giá trị cao, màu đen nếu giá trị thấp
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Hiển thị số liệu trên ô
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_loss_curve(losses, title="Training Loss"):
    """
    Vẽ đường cong loss theo từng epoch trong quá trình training.
    
    Args:
        losses (list or np.ndarray): Danh sách các giá trị loss qua từng epoch, length = số epochs
        title (str): Tiêu đề của biểu đồ. Mặc định: "Training Loss"
    
    Returns:
        None: Hiển thị biểu đồ trực tiếp
    
    Notes:
        - Trục x: Số epoch (bắt đầu từ 0)
        - Trục y: Giá trị loss
        - Đường loss giảm dần cho thấy mô hình đang học tốt
    """
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()