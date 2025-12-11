"""
MNIST Digit Recognition App
===========================
Ứng dụng nhận dạng chữ số viết tay sử dụng PCA + SoftmaxRegression

Features:
- Vẽ chữ số bằng chuột
- Tải ảnh từ file (JPG, PNG, etc.)
- Demo với ảnh từ test set
- Hiển thị xác suất dự đoán
- Giao diện đẹp, trực quan

Author: Your Name
Date: December 2025
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# Thêm đường dẫn thư viện
sys.path.insert(0, './lib')
from SoftmaxRegression import SoftmaxRegression

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        font-size: 4rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        text-align: center;
        color: #333;
        margin-top: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load PCA model và trained weights"""
    model_path = './models/best_model_weights.npz'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model tại: {model_path}")
        st.stop()
    
    # Load model data
    model_data = np.load(model_path)
    pca_mean = model_data['pca_mean']
    pca_vt = model_data['pca_vt']
    pca_n_components = int(model_data['pca_n_components'])
    model_weights = model_data['model_weights']
    model_bias = model_data['model_bias']
    
    # Reconstruct model
    model = SoftmaxRegression(
        n_features=pca_n_components,
        n_classes=10,
        learning_rate=0.05
    )
    model.W = model_weights
    model.b = model_bias
    
    return {
        'model': model,
        'pca_mean': pca_mean,
        'pca_vt': pca_vt,
        'n_components': pca_n_components
    }


@st.cache_data
def load_test_data():
    """Load MNIST test data"""
    data_path = './data/raw/mnist.npz'
    if not os.path.exists(data_path):
        return None, None
    
    data = np.load(data_path)
    X_test = data['x_test'] / 255.0  # Normalize
    y_test = data['y_test']
    return X_test, y_test


# ==================== HELPER FUNCTIONS ====================
def preprocess_image(img_array, target_size=28):
    """Preprocess image: resize to 28x28 and normalize"""
    # Convert to grayscale if RGB
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        img_gray = np.mean(img_array, axis=2)
    else:
        img_gray = img_array
    
    # Resize to 28x28
    img_pil = Image.fromarray((img_gray * 255).astype(np.uint8))
    img_pil = img_pil.resize((target_size, target_size), Image.LANCZOS)
    img_resized = np.array(img_pil) / 255.0
    
    return img_resized


def apply_pca(img_flat, pca_mean, pca_vt):
    """Apply PCA transformation"""
    img_centered = img_flat - pca_mean
    img_pca = img_centered @ pca_vt.T
    return img_pca


def predict_digit(img_array, model_dict):
    """
    Predict digit from image
    
    Returns:
        dict with prediction, confidence, probabilities, preprocessed_image
    """
    # Preprocess
    img_processed = preprocess_image(img_array)
    img_flat = img_processed.flatten()
    
    # Apply PCA
    img_pca = apply_pca(img_flat, model_dict['pca_mean'], model_dict['pca_vt'])
    
    # Predict
    y_proba = model_dict['model'].predict_proba(img_pca.reshape(1, -1))[0]
    y_pred = np.argmax(y_proba)
    confidence = y_proba[y_pred]
    
    return {
        'prediction': y_pred,
        'confidence': confidence,
        'probabilities': y_proba,
        'preprocessed_image': img_processed
    }


def plot_probabilities(probabilities, prediction):
    """Plot probability bar chart"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['#e74c3c' if i == prediction else '#3498db' for i in range(10)]
    bars = ax.bar(range(10), probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Highlight prediction
    bars[prediction].set_color('#2ecc71')
    bars[prediction].set_alpha(1.0)
    
    ax.set_xlabel('Chữ Số (0-9)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Xác Suất', fontsize=12, fontweight='bold')
    ax.set_title('Phân Bố Xác Suất Dự Đoán', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xticks(range(10))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels on top of bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob*100:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">🔢 MNIST Digit Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Nhận dạng chữ số viết tay với PCA + SoftmaxRegression</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('⏳ Đang tải mô hình...'):
        model_dict = load_model()
    
    st.success('✅ Mô hình đã sẵn sàng!')
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Thông Tin Mô Hình")
        st.markdown("---")
        
        st.metric("Phương pháp", "PCA + Softmax")
        st.metric("PCA Components", f"{model_dict['n_components']}")
        st.metric("Độ chính xác", "91.82%")
        st.metric("Thời gian train", "~16 giây")
        
        st.markdown("---")
        st.header("ℹ️ Hướng Dẫn")
        st.markdown("""
        **3 cách sử dụng:**
        1. 🎨 **Vẽ chữ số** - Dùng chuột vẽ trực tiếp
        2. 📁 **Tải ảnh** - Upload file từ máy
        3. 🎲 **Demo MNIST** - Xem ảnh mẫu từ test set
        
        **Tips:**
        - Vẽ chữ số to, rõ ràng
        - Nền đen, nét trắng (hoặc ngược lại)
        - Ảnh upload nên là ảnh đơn giản
        """)
        
        st.markdown("---")
        st.info("💡 **Model Info:**\n\n"
                "- Features: 784 → 80 (PCA)\n"
                "- Giảm chiều: 89.8%\n"
                "- Variance giữ lại: 89.06%\n"
                "- Batch size: 32\n"
                "- Learning rate: 0.05")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎨 Vẽ Chữ Số", "📁 Tải Ảnh Lên", "🎲 Demo MNIST"])
    
    # ========== TAB 1: DRAWING CANVAS ==========
    with tab1:
        st.header("🎨 Vẽ Chữ Số Bằng Chuột")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Canvas - Vẽ tại đây")
            
            # Canvas settings
            stroke_width = st.slider("Độ dày nét vẽ", 10, 50, 25)
            
            # Drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_btn = st.button("🔮 Dự Đoán", use_container_width=True)
            with col_btn2:
                clear_btn = st.button("🗑️ Xóa Canvas", use_container_width=True)
        
        with col2:
            st.subheader("Kết Quả Dự Đoán")
            
            if canvas_result.image_data is not None and predict_btn:
                # Get canvas image
                img_data = canvas_result.image_data
                
                # Check if canvas has drawing
                if img_data[:, :, 3].max() > 0:  # Check alpha channel
                    # Convert RGBA to grayscale
                    img_gray = np.mean(img_data[:, :, :3], axis=2) / 255.0
                    
                    # Predict
                    result = predict_digit(img_gray, model_dict)
                    
                    # Display prediction
                    st.markdown(f'<div class="prediction-box">Dự đoán: {result["prediction"]}</div>', 
                              unsafe_allow_html=True)
                    st.markdown(f'<p class="confidence-text">Độ tin cậy: {result["confidence"]*100:.2f}%</p>', 
                              unsafe_allow_html=True)
                    
                    # Show preprocessed image
                    st.subheader("Ảnh sau xử lý (28×28)")
                    fig_img, ax_img = plt.subplots(figsize=(4, 4))
                    ax_img.imshow(result['preprocessed_image'], cmap='gray')
                    ax_img.axis('off')
                    st.pyplot(fig_img)
                    plt.close()
                    
                    # Probability chart
                    st.subheader("Phân Bố Xác Suất")
                    fig_prob = plot_probabilities(result['probabilities'], result['prediction'])
                    st.pyplot(fig_prob)
                    plt.close()
                    
                    # Top 3 predictions
                    st.subheader("🏆 Top 3 Dự Đoán")
                    top_3_idx = np.argsort(result['probabilities'])[-3:][::-1]
                    for i, idx in enumerate(top_3_idx, 1):
                        st.write(f"{i}. Chữ số **{idx}**: {result['probabilities'][idx]*100:.2f}%")
                else:
                    st.warning("⚠️ Canvas trống! Vui lòng vẽ chữ số trước.")
    
    # ========== TAB 2: FILE UPLOAD ==========
    with tab2:
        st.header("📁 Tải Ảnh Chữ Số Từ File")
        
        uploaded_file = st.file_uploader(
            "Chọn ảnh chữ số (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            help="Tải lên ảnh chữ số viết tay để nhận dạng"
        )
        
        if uploaded_file is not None:
            # Read image
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Ảnh Gốc")
                st.image(img, use_container_width=True)
                st.caption(f"Kích thước: {img_array.shape}")
            
            with col2:
                st.subheader("Kết Quả Dự Đoán")
                
                # Predict
                result = predict_digit(img_array, model_dict)
                
                # Display prediction
                st.markdown(f'<div class="prediction-box">Dự đoán: {result["prediction"]}</div>', 
                          unsafe_allow_html=True)
                st.markdown(f'<p class="confidence-text">Độ tin cậy: {result["confidence"]*100:.2f}%</p>', 
                          unsafe_allow_html=True)
                
                # Preprocessed image
                st.subheader("Ảnh sau xử lý (28×28)")
                fig_img, ax_img = plt.subplots(figsize=(4, 4))
                ax_img.imshow(result['preprocessed_image'], cmap='gray')
                ax_img.axis('off')
                st.pyplot(fig_img)
                plt.close()
            
            # Full width probability chart
            st.subheader("Phân Bố Xác Suất")
            fig_prob = plot_probabilities(result['probabilities'], result['prediction'])
            st.pyplot(fig_prob)
            plt.close()
            
            # Detailed probabilities
            st.subheader("📊 Chi Tiết Xác Suất")
            for i in range(10):
                st.progress(result['probabilities'][i], 
                          text=f"Chữ số {i}: {result['probabilities'][i]*100:.2f}%")
    
    # ========== TAB 3: MNIST DEMO ==========
    with tab3:
        st.header("🎲 Demo Với MNIST Test Set")
        
        # Load test data
        X_test, y_test = load_test_data()
        
        if X_test is not None:
            st.info(f"📦 Test set: {X_test.shape[0]} ảnh sẵn có")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Random sample button
                if st.button("🎲 Chọn Ảnh Ngẫu Nhiên", use_container_width=True):
                    st.session_state.random_idx = np.random.randint(0, len(X_test))
                
                # Or select specific index
                img_idx = st.number_input(
                    "Hoặc chọn index (0-9999)",
                    min_value=0,
                    max_value=len(X_test)-1,
                    value=st.session_state.get('random_idx', 0),
                    step=1
                )
                
                # Display selected image
                st.subheader("Ảnh Được Chọn")
                selected_img = X_test[img_idx]
                true_label = y_test[img_idx]
                
                fig_orig, ax_orig = plt.subplots(figsize=(4, 4))
                ax_orig.imshow(selected_img, cmap='gray')
                ax_orig.set_title(f'Nhãn thực: {true_label}', fontsize=12, fontweight='bold')
                ax_orig.axis('off')
                st.pyplot(fig_orig)
                plt.close()
            
            with col2:
                # Predict
                result = predict_digit(selected_img, model_dict)
                
                # Display prediction
                st.subheader("Kết Quả Dự Đoán")
                
                is_correct = result['prediction'] == true_label
                color = "green" if is_correct else "red"
                status = "✅ ĐÚNG" if is_correct else "❌ SAI"
                
                st.markdown(f'<div class="prediction-box" style="background: linear-gradient(135deg, '
                          f'{"#27ae60" if is_correct else "#c0392b"} 0%, '
                          f'{"#229954" if is_correct else "#922b21"} 100%);">'
                          f'Dự đoán: {result["prediction"]}</div>', 
                          unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Nhãn thực", true_label)
                with col_b:
                    st.metric("Kết quả", status)
                
                st.markdown(f'<p class="confidence-text">Độ tin cậy: {result["confidence"]*100:.2f}%</p>', 
                          unsafe_allow_html=True)
                
                # Probability chart
                st.subheader("Phân Bố Xác Suất")
                fig_prob = plot_probabilities(result['probabilities'], result['prediction'])
                st.pyplot(fig_prob)
                plt.close()
        else:
            st.error("❌ Không tìm thấy MNIST test data tại ./data/raw/mnist.npz")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem;'>
        <p>🎓 <b>MNIST Digit Recognition App</b></p>
        <p>Sử dụng PCA (80 components) + SoftmaxRegression</p>
        <p>Accuracy: 91.82% | Runtime: ~16s | Features: 784 → 80</p>
        <p style='margin-top: 1rem;'>Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
