import numpy as np
import os

class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.1):
        """
        Khởi tạo mô hình Softmax Regression cơ bản.

        Args:
            n_features (int): Số lượng đặc trưng đầu vào (ví dụ: 784).
            n_classes (int): Số lượng lớp đầu ra (ví dụ: 10).
            learning_rate (float): Tốc độ học (alpha).
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.losses = []
        
        # Khởi tạo trọng số (W) và bias (b)
        # Sử dụng phân phối chuẩn nhỏ để khởi tạo W
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))

    def softmax(self, z):
        """
        Tính toán Softmax với ổn định số học (Numerical Stability).
        Formula: exp(z_i) / sum(exp(z_j))
        """
        # Trừ max(z) để tránh tràn số mũ (overflow)
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Lan truyền xuôi (Forward pass): Z = XW + b -> Softmax(Z).
        """
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)

    def compute_loss(self, y_true, y_pred):
        """
        Tính hàm mất mát Cross-Entropy.
        Cost = - (1/m) * Sum(y_true * log(y_pred))
        """
        m = y_true.shape[0]
        epsilon = 1e-9 # Giá trị nhỏ để tránh lỗi log(0)
        
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        return loss

    def backward(self, X, y_true, y_pred):
        """
        Tính Gradient (đạo hàm) của hàm Loss theo W và b.
        """
        m = X.shape[0]
        
        # Sai số dự đoán: dZ = A - Y
        dz = y_pred - y_true
        
        # Gradient của W: dW = (1/m) * X.T . dZ
        dw = np.dot(X.T, dz) / m
        
        # Gradient của b: db = (1/m) * sum(dZ)
        db = np.sum(dz, axis=0, keepdims=True) / m
        
        return dw, db

    def fit(self, X, y, epochs=100, batch_size=256, verbose=True):
        """
        Huấn luyện mô hình bằng Mini-batch Gradient Descent.
        """
        m = X.shape[0]
        self.losses = [] # Reset lịch sử loss

        for epoch in range(epochs):
            # Xáo trộn dữ liệu (Shuffle)
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            num_batches = int(np.ceil(m / batch_size))

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, m)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # 1. Forward
                y_pred = self.forward(X_batch)

                # 2. Loss (Tính tổng loss để hiển thị)
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss * (end_idx - start_idx)

                # 3. Backward
                dw, db = self.backward(X_batch, y_batch, y_pred)

                # 4. Update parameters (Basic Gradient Descent)
                # W = W - learning_rate * dW
                self.W -= self.lr * dw
                self.b -= self.lr * db
            
            # Tính loss trung bình cho cả epoch
            avg_loss = epoch_loss / m
            self.losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def predict(self, X):
        """Dự đoán nhãn lớp (trả về chỉ số lớp có xác suất cao nhất)."""
        y_pred_probs = self.forward(X)
        return np.argmax(y_pred_probs, axis=1)
    
    def predict_proba(self, X):
        """Trả về xác suất dự đoán."""
        return self.forward(X)

    def save_weights(self, filepath):
        """Lưu trọng số W và b vào file .npz."""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        np.savez(filepath, W=self.W, b=self.b)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        """Tải trọng số W và b từ file .npz."""
        if not os.path.exists(filepath):
            print(f"File {filepath} not found.")
            return False
        data = np.load(filepath)
        self.W = data['W']
        self.b = data['b']
        print(f"Weights loaded from {filepath}")
        return True