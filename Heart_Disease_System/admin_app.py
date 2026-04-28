import warnings
warnings.simplefilter("ignore")
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io
import google.generativeai as genai
from scipy import stats
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
# --- THÊM DÒNG NÀY ---
try:
    from xgboost import XGBClassifier
except ImportError:
    st.error("Thiếu thư viện 'xgboost'. Vui lòng chạy lệnh: pip install xgboost")
    st.stop()
import matplotlib.pyplot as plt
try:
    import shap
except ImportError:
    st.warning("Thiếu thư viện 'shap'. Một số tính năng giải thích nâng cao sẽ bị tắt. Cài đặt: pip install shap")
# Thư viện Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve 
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, roc_curve, auc, precision_score, recall_score,
    precision_recall_curve, average_precision_score, matthews_corrcoef
)
from sklearn.impute import SimpleImputer, KNNImputer

# Thư viện xử lý mất cân bằng dữ liệu
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    st.error("Thiếu thư viện 'imbalanced-learn'. Vui lòng chạy lệnh: pip install imbalanced-learn")
    st.stop()

# =========================================================
# 1. CẤU HÌNH TRANG (GIỮ NGUYÊN FULL)
# =========================================================
st.set_page_config(
    page_title="Heart Disease Diagnosis System - SVM Pro",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
px.defaults.template = "plotly_white"

# Custom CSS giao diện (GIỮ NGUYÊN)
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #e63946;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #1d3557; }
    .metric-label { font-size: 13px; color: #457b9d; font-weight: 600; text-transform: uppercase; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .highlight-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #90caf9;
    }
    .stDataFrame { border: 1px solid #ddd; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 2. HÀM XỬ LÝ DỮ LIỆU
# =========================================================
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None

def clean_data(df):
    df_clean = df.copy()
    
    # Loại bỏ cột định danh không cần thiết
    cols_to_drop = ['SNO', 'MRD No.', 'D.O.A', 'D.O.D', 'month year', 'Name', 'ID', 'Patient_ID']
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns])
    
    # Ép kiểu số
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            temp = pd.to_numeric(df_clean[col], errors='coerce')
            if temp.isna().mean() < 0.9:  
                df_clean[col] = temp

    # Feature Engineering (Tạo đặc trưng mới)
    sys_cols = [c for c in df_clean.columns if 'systolic' in c.lower() or 'sbp' in c.lower()]
    dia_cols = [c for c in df_clean.columns if 'diastolic' in c.lower() or 'dbp' in c.lower()]
    
    if sys_cols and dia_cols:
        try:
            s_col = sys_cols[0]
            d_col = dia_cols[0]
            df_clean[s_col] = pd.to_numeric(df_clean[s_col], errors='coerce')
            df_clean[d_col] = pd.to_numeric(df_clean[d_col], errors='coerce')
            df_clean['Pulse_Pressure'] = df_clean[s_col] - df_clean[d_col]
        except:
            pass 

    # --- LƯU Ý: KHÔNG TỰ ĐỘNG ĐIỀN SỐ LIỆU (ĐỂ XỬ LÝ Ở TAB 1 THEO YÊU CẦU) ---
    
    # Mã hóa Label Encoding cho biến phân loại
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        # Xử lý NaN cho biến Categorical để tránh lỗi code, dùng mode
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            
        df_clean[col] = df_clean[col].astype(str)
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        le_dict[col] = le
        
    return df_clean, le_dict

# --- HÀM MỚI BỔ SUNG: VẼ LEARNING CURVE ---
def plot_learning_curve_graph(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()
    
    # Training Score
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_scores_mean, 
        mode='lines+markers', name='Training Score', 
        line=dict(color='blue'), 
        error_y=dict(type='data', array=train_scores_std, visible=True)
    ))
    
    # Cross-validation Score
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_scores_mean, 
        mode='lines+markers', name='Cross-Validation Score', 
        line=dict(color='green'),
        error_y=dict(type='data', array=test_scores_std, visible=True)
    ))
    
    fig.update_layout(
        title="Learning Curve (Đường cong học tập)", 
        xaxis_title="Số lượng mẫu Training", 
        yaxis_title="Accuracy Score",
        template="plotly_white"
    )
    return fig

# =========================================================
# 3. MÀN HÌNH GIỚI THIỆU (GIỮ NGUYÊN)
# =========================================================
def show_intro_page():
    st.markdown("<h1 style='text-align: center; color: #D32F2F;'>❤️ HỆ THỐNG CHẨN ĐOÁN BỆNH SUY TIM (SVM)</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 📌 Chức năng hệ thống:
        
        ✅ **Khám phá dữ liệu (EDA):** Phân tích thống kê, biểu đồ phân phối và **AI chọn lọc đặc trưng**.
        
        ✅ **Huấn luyện AI (SVM):** Sử dụng thuật toán Support Vector Machine với tối ưu hóa Kernel và cân bằng dữ liệu (SMOTE).
        
        ✅ **Đánh giá chuyên sâu (Medical Metrics):** Sensitivity, Specificity, PPV, NPV, Learning Curve.
        
        ✅ **Dự đoán thông minh:** Hỗ trợ nhập liệu nhanh (Quick Predict) dựa trên các chỉ số quan trọng nhất.
        """)
        
        st.info("💡 **Gợi ý:** File dữ liệu cần định dạng `.csv` và chứa các chỉ số lâm sàng (Tuổi, Huyết áp, EF, BNP, v.v.)")
    
    with col2:
        st.markdown("""
        ### 🚀 Quy trình vận hành (Workflow):
        
        1.  **Tải dữ liệu:** Upload file `.csv` ở thanh bên trái (Sidebar).
        2.  **Xử lý & AI:** (Tab 1) Tự động xử lý dữ liệu thiếu, mã hóa và dùng **Gemini AI** để phân tích ý nghĩa.
        3.  **Trực quan hóa:** (Tab 2) Xem Dashboard phân bố, tương quan và biểu đồ 3D.
        4.  **Huấn luyện & So sánh:** (Tab 3) Chạy đa mô hình (SVM, RF, KNN...), tối ưu tham số (Auto-tune) và xem **Bảng xếp hạng (Leaderboard)**.
        5.  **Chọn mô hình triển khai:** (Tab 4) Chọn mô hình để triển khai chẩn đoán
        
        ### 🛠️ Công nghệ cốt lõi (Tech Stack):
        * **Nền tảng:** Python & Streamlit Framework.
        * **Machine Learning:** Scikit-learn (SVM, Random Forest, Logistic Regression, KNN).
        * **Xử lý dữ liệu:** Pandas, NumPy, Imbalanced-learn (SMOTE, ADASYN).
        * **Trực quan hóa:** Plotly Express (Interactive), Matplotlib.
        * **Explainable AI (XAI):** Permutation Importance.
        * **Generative AI:** Google Gemini API.
        """)

# =========================================================
# 4. GIAO DIỆN CHÍNH (MAIN APP)
# =========================================================
def main():
    # --- SIDEBAR: CHỈ CÒN UPLOAD FILE ---
    st.sidebar.title("⚙️ Điều khiển")
    st.sidebar.subheader("1. Nguồn dữ liệu")
    uploaded_file = st.sidebar.file_uploader("Upload file CSV:", type=['csv'])
    
    # === KIỂM TRA FILE ===
    if uploaded_file is None:
        show_intro_page()
        # Reset session khi chưa có file
        if 'df_main' in st.session_state:
            del st.session_state['df_main']
        return

    # === XỬ LÝ DỮ LIỆU (QUẢN LÝ BẰNG SESSION STATE) ===
    file_id = f"file_{uploaded_file.name}_{uploaded_file.size}"
    
    if 'current_file_id' not in st.session_state or st.session_state['current_file_id'] != file_id:
        with st.spinner('Đang tải và xử lý sơ bộ dữ liệu...'):
            df_raw_loaded = load_data(uploaded_file)
            if df_raw_loaded is None: return
            df_clean_loaded, le_dict = clean_data(df_raw_loaded)
            
            # Lưu vào Session State
            st.session_state['df_main'] = df_clean_loaded
            st.session_state['df_raw'] = df_raw_loaded
            st.session_state['le_dict'] = le_dict
            st.session_state['current_file_id'] = file_id
            # Reset biến AI gợi ý khi load file mới
            if 'ai_suggested_features' in st.session_state:
                del st.session_state['ai_suggested_features']
            # Reset trạng thái train
            st.session_state['is_trained'] = False
    
    # Lấy dữ liệu từ Session State để sử dụng
    df_clean = st.session_state['df_main']
    df_raw = st.session_state['df_raw']

    st.sidebar.success(f"✅ Đã tải: {len(df_clean)} hồ sơ")
    
    # Cảnh báo nếu còn dữ liệu thiếu
    missing_count = df_clean.isnull().sum().sum()
    if missing_count > 0:
        st.sidebar.warning(f"⚠️ Dữ liệu còn {missing_count} giá trị thiếu! Vui lòng xử lý ở Tab 1.")
    
    st.sidebar.info("👉 Hãy chuyển sang Tab 'Cấu hình & Huấn luyện' để chọn biến mục tiêu.")
    st.sidebar.markdown("---")

    # --- MAIN TABS ---
    st.title("📊 Phân tích & Chẩn đoán Bệnh Suy Tim")
    
    # Tạo 4 Tab như cũ
    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Dữ liệu & AI Phân tích", 
        "📈 Thống kê và Trực quan hóa (EDA) dữ liệu ", 
        "🧠 Cấu hình & Huấn luyện ", 
        "🔍 Dự đoán"
    ])


    # ==================== TAB 1: DATA & GEMINI AI  ====================
    with tab1:
        st.subheader("Tổng quan dữ liệu")
        st.dataframe(df_raw.head())
        
        c1, c2 = st.columns(2)
        c1.metric("Số lượng mẫu", df_clean.shape[0])
        c2.metric("Số lượng cột", df_clean.shape[1])
        
        # --- PHẦN XỬ LÝ DỮ LIỆU THIẾU  ---
        st.markdown("---")
        st.subheader("🛠️ Phân tích & Xử lý Dữ liệu thiếu")
        
        # Tính toán lượng dữ liệu thiếu
        missing_data = df_clean.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        col_miss1, col_miss2 = st.columns([2, 1])
        
        # --- TRƯỜNG HỢP 1: CÓ DỮ LIỆU THIẾU ---
        if not missing_data.empty:
            with col_miss1:
                fig_missing = px.bar(
                    x=missing_data.index, 
                    y=missing_data.values,
                    labels={'x': 'Tên cột', 'y': 'Số lượng thiếu'},
                    title=f"Biểu đồ phân bố giá trị thiếu (Tổng: {missing_data.sum()})",
                    color=missing_data.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_missing, use_container_width=True)
                
            with col_miss2:
                st.dataframe(missing_data, use_container_width=True)

            st.markdown("### ⚡ Tùy chọn Xử lý:")
            
            # 1. Xác định Top 3 cột thiếu nhiều nhất
            all_missing_cols = missing_data.index.tolist()
            top_3_cols = all_missing_cols[:3]
            rest_cols = all_missing_cols[3:]
            
            cols_to_drop = [] 
            
            # Giao diện cho Top 3
            st.warning("⚠️ **Top 3 cột thiếu dữ liệu nhiều nhất:**")
            st.caption("Bạn có thể chọn **Xóa (Drop)** để giảm chiều dữ liệu hoặc **Giữ lại** để điền bù bằng KNN.")
            
            for col in top_3_cols:
                missing_count = missing_data[col]
                pct = (missing_count / len(df_clean)) * 100
                
                # Checkbox: Nếu tích vào thì XÓA, không tích thì dùng KNN
                check_label = f"🗑️ Xóa cột **'{col}'** (Thiếu {missing_count} dòng - {pct:.1f}%)"
                if st.checkbox(check_label, value=False, key=f"chk_drop_{col}"):
                    cols_to_drop.append(col)
                else:
                    st.write(f"   ↳ *Giữ lại và xử lý bằng KNN Imputer*")

            # Giao diện cho các cột còn lại
            if rest_cols:
                st.info(f"ℹ️ **Các cột thiếu ít dữ liệu ({len(rest_cols)} cột):** {', '.join(rest_cols)}. \n\n 👉 Hệ thống sẽ tự động giữ lại và xử lý bằng **KNN Imputer**.")

            st.markdown("---")
            
            # Nút thực hiện xử lý
            if st.button("🔧 Tiến hành Xử lý Dữ liệu", type="primary"):
                with st.spinner("Đang xử lý dữ liệu (Xóa cột + KNN Imputer)..."):
                    # B1. Xóa cột được chọn
                    if cols_to_drop:
                        df_clean = df_clean.drop(columns=cols_to_drop)
                    
                    # B2. Xử lý KNN cho các cột còn lại
                    num_cols = df_clean.select_dtypes(include=[np.number]).columns
                    if df_clean[num_cols].isnull().sum().sum() > 0:
                        imputer = KNNImputer(n_neighbors=5, weights='uniform')
                        df_filled_vals = imputer.fit_transform(df_clean[num_cols])
                        df_clean[num_cols] = df_filled_vals
                    
                    # B3. Lưu và Rerun
                    st.session_state['df_main'] = df_clean
                    st.success("🎉 Xử lý hoàn tất! Đang làm mới trang...")
                    st.rerun()

        # --- TRƯỜNG HỢP 2: DỮ LIỆU ĐÃ SẠCH  ---
        else:
            with col_miss1:
                st.success("✅ Tuyệt vời! Dữ liệu sạch sẽ, không có giá trị nào bị thiếu.")
            
            # --- DOWNLOAD FILE ĐÃ XỬ LÝ ---
            st.markdown("---")
            st.subheader("📥 Tải xuống dữ liệu đã làm sạch")
            st.caption("Bạn có thể tải file CSV này về để lưu trữ hoặc sử dụng cho các phần mềm khác.")
            
            # Chuyển đổi DataFrame sang CSV
            csv_processed = df_clean.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="⬇️ Tải file CSV sạch (.csv)",
                data=csv_processed,
                file_name="cleaned_heart_data.csv",
                mime="text/csv",
                type="primary",
                key="download_clean_csv"
            )

        # --- PHẦN XỬ LÝ MÃ HÓA (LABEL ENCODER) ---
        # Tự động tìm cột chữ để chuyển sang số và LƯU LẠI ENCODER
        object_cols = df_clean.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            # Khởi tạo dict lưu trữ nếu chưa có
            if 'le_dict' not in st.session_state:
                st.session_state['le_dict'] = {}
                
            from sklearn.preprocessing import LabelEncoder
            for col in object_cols:
                # Nếu chưa mã hóa thì mới làm
                if df_clean[col].dtype == 'object':
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                    # Lưu encoder vào session để Tab 3 và Tab 4 dùng
                    st.session_state['le_dict'][col] = le
            
            # Cập nhật lại dataframe
            st.session_state['df_main'] = df_clean

        st.markdown("---")
        
        # --- PHẦN GEMINI AI (GIỮ NGUYÊN) ---
        st.subheader("🤖 AI Giải thích & Gợi ý Đặc trưng (Gemini)")
        st.caption("AI sẽ giải thích ý nghĩa các cột và **đánh dấu các cột quan trọng nhất ** (Đã lọc bỏ biến kết quả).")
        
        GEMINI_API_KEY = "YOUR API KEY" 
        
        should_show_analysis = st.button("✨ Phân tích & Tìm đặc trưng quan trọng", key="gemini_eda_btn") or ('gemini_result_json' in st.session_state)

        if should_show_analysis:
            if 'gemini_result_json' not in st.session_state:
                try:
                    with st.spinner("Gemini đang đọc dữ liệu, phân tích y khoa và lọc đặc trưng..."):
                        genai.configure(api_key=GEMINI_API_KEY)
                        model = genai.GenerativeModel('models/gemini-2.0-flash')
                        column_list = ", ".join(df_clean.columns.tolist())
                        
                        prompt = f"""
                        Bạn là chuyên gia y tế tim mạch. Dữ liệu có các cột: {column_list}.
                        Nhiệm vụ:
                        1. Giải thích ý nghĩa lâm sàng ngắn gọn từng cột.
                        2. Chọn các cột QUAN TRỌNG NHẤT (Input Features) để dự báo bệnh.(Tối đa 12 cột)
                        QUAN TRỌNG: KHÔNG chọn 'Outcome', 'Death', 'Alive', 'Target'.
                        OUTPUT JSON: {{ "data_dictionary": [{{"Tên cột": "Age", "Ý nghĩa lâm sàng": "..."}}], "important_features": ["Age", "EF"] }}
                        """
                        response = model.generate_content(prompt)
                        text_resp = response.text.strip().replace("```json", "").replace("```", "")
                        result_json = json.loads(text_resp)
                        st.session_state['gemini_result_json'] = result_json
                except Exception as e:
                    st.error(f"⚠️ Lỗi kết nối Gemini API: {e}")

            if 'gemini_result_json' in st.session_state:
                result_json = st.session_state['gemini_result_json']
                data_dict_list = result_json.get("data_dictionary", [])
                desc_map = {item['Tên cột']: item.get('Ý nghĩa lâm sàng', 'Chưa có mô tả') for item in data_dict_list}
                
                df_ai = pd.DataFrame(data_dict_list)
                st.dataframe(df_ai, use_container_width=True, hide_index=True)
                
                imp_feats = result_json.get("important_features", [])
                forbidden_keywords = ['outcome', 'death', 'alive', 'survived', 'mortality', 'target', 'label', 'event']
                valid_imp_feats = [col for col in imp_feats if col in df_clean.columns and not any(k in col.lower() for k in forbidden_keywords)]
                
                if valid_imp_feats:
                    st.session_state['ai_suggested_features'] = valid_imp_feats
                    st.success(f"✅ Gemini đã chọn {len(valid_imp_feats)} đặc trưng quan trọng.")
                    cols_show = st.columns(4)
                    for idx, feat in enumerate(valid_imp_feats):
                        with cols_show[idx % 4]:
                            st.metric(label="Đặc trưng", value=feat, help=desc_map.get(feat, ""))
                            st.write("")
                else:
                    st.warning("Không tìm thấy đặc trưng phù hợp.")

        st.markdown("---")
        st.subheader("Thống kê mô tả")
        st.dataframe(df_clean.describe())
# ==================== TAB 2: VISUALIZATION (DASHBOARD FINAL - ĐÃ SỬA LỖI) ====================
    with tab2:
        # --- LẤY DỮ LIỆU ---
        if 'df_main' in st.session_state:
            df_clean = st.session_state['df_main']
            
        st.header("📈 Dashboard Phân tích Dữ liệu Y tế")
        st.caption("Tổng quan về dữ liệu bệnh nhân, phân bố bệnh lý và các yếu tố nguy cơ tiềm ẩn.")

        # --- 1. TỰ ĐỘNG TÌM TARGET ĐỂ HIỂN THỊ DASHBOARD ---
        priority_targets = ['HEART FAILURE', 'ACS', 'STEMI', 'TARGET', 'OUTPUT', 'DEATH_EVENT', 'Label', 'Outcome', 'DEATH']
        viz_target_idx = 0
        viz_target = None
        for t in priority_targets:
            if t in df_clean.columns:
                viz_target = t
                viz_target_idx = list(df_clean.columns).index(t)
                break
        
        # --- 2. KPI DASHBOARD (METRICS) ---
        st.subheader("📌 Tổng quan Số liệu (Key Metrics)")
        if viz_target:
            total_patients = len(df_clean)
            num_positive = df_clean[viz_target].sum()
            positive_rate = (num_positive / total_patients) * 100
            num_negative = total_patients - num_positive
            
            # Hiển thị 4 thẻ Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng số hồ sơ", f"{total_patients:,}", delta="Dữ liệu gốc")
            m2.metric("Ca Dương tính (Bệnh)", f"{num_positive:,}", f"{positive_rate:.1f}%")
            m3.metric("Ca Âm tính (Khỏe)", f"{num_negative:,}", f"{100-positive_rate:.1f}%")
            
            # Thẻ 4: Tuổi trung bình (nếu có) hoặc số đặc trưng
            age_col = next((c for c in df_clean.columns if c.upper() == 'AGE'), None)
            if age_col:
                avg_age = df_clean[age_col].mean()
                m4.metric("Tuổi trung bình", f"{avg_age:.1f}", "Năm")
            else:
                m4.metric("Số lượng đặc trưng", f"{df_clean.shape[1]}")

        st.markdown("---")

        # --- 3. CẤU HÌNH & BIỂU ĐỒ TỔNG QUAN ---
        col_viz_sel1, col_viz_sel2 = st.columns([1, 3])
        with col_viz_sel1:
            # Chọn biến target (Cho phép người dùng đổi nếu muốn)
            viz_target_select = st.selectbox("🎯 Target (Phân nhóm):", df_clean.columns, index=viz_target_idx, key="viz_target_main")
        with col_viz_sel2:
            st.info(f"Đang phân tích dữ liệu dựa trên việc phân nhóm theo: **{viz_target_select}**")

        c1, c2 = st.columns(2)
        with c1:
            # Pie Chart
            counts = df_clean[viz_target_select].value_counts().reset_index()
            counts.columns = ['Label', 'Count']
            fig_pie = px.pie(counts, values='Count', names='Label', hole=0.4, 
                             title=f"Tỷ lệ phân bố nhóm: {viz_target_select}", 
                             color_discrete_sequence=px.colors.qualitative.Safe)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            # Bar Chart Correlation
            num_cols = df_clean.select_dtypes(include=[np.number])
            if viz_target_select in num_cols.columns:
                corr_vals = num_cols.corrwith(df_clean[viz_target_select]).sort_values(ascending=False)
                corr_vals = corr_vals.drop(viz_target_select) # Bỏ chính nó
                
                fig_corr = px.bar(x=corr_vals.values, y=corr_vals.index, orientation='h',
                                  title=f"Mức độ tương quan với {viz_target_select}",
                                  labels={'x': 'Hệ số tương quan', 'y': 'Đặc trưng'},
                                  color=corr_vals.values, color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)

        # --- 4. PHÂN TÍCH CHI TIẾT (TABS) ---
        st.markdown("---")
        st.subheader("🔍 Phân tích Chi tiết từng Đặc trưng")
        
        # Tách biến
        numeric_vars = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        numeric_vars = [c for c in numeric_vars if c != viz_target_select]
        cat_vars = [c for c in df_clean.columns if df_clean[c].nunique() < 10 and c != viz_target_select]

        tab_num, tab_cat, tab_3d = st.tabs(["📊 Biến Định lượng (Số)", "📌 Biến Định danh (Phân loại)", "🧊 Không gian 3D"])
        
        # --- TAB 1: SỐ HỌC ---
        with tab_num:
            if numeric_vars:
                feat_viz = st.selectbox("Chọn chỉ số số học:", numeric_vars, key="num_viz")
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    fig_hist = px.histogram(df_clean, x=feat_viz, color=viz_target_select, 
                                            marginal="box", barmode="overlay", opacity=0.7,
                                            title=f"Phân phối {feat_viz}", color_discrete_sequence=px.colors.qualitative.Set1)
                    st.plotly_chart(fig_hist, use_container_width=True)
                with c_d2:
                    fig_vio = px.violin(df_clean, y=feat_viz, x=viz_target_select, color=viz_target_select, 
                                        box=True, points="all", title=f"Mật độ & Outlier: {feat_viz}")
                    st.plotly_chart(fig_vio, use_container_width=True)
                
                # Auto Insights
                try:
                    means = df_clean.groupby(viz_target_select)[feat_viz].mean()
                    diff = abs(means.iloc[0] - means.iloc[1])
                    st.info(f"💡 **Nhận xét:** Chênh lệch trung bình giữa 2 nhóm là **{diff:.2f}**.")
                except: pass
            else: st.warning("Không có biến số.")

        # --- TAB 2: PHÂN LOẠI ---
        with tab_cat:
            if cat_vars:
                cat_feat = st.selectbox("Chọn biến phân loại:", cat_vars, key="cat_viz")
                ct = pd.crosstab(df_clean[cat_feat], df_clean[viz_target_select], normalize='index') * 100
                ct_melt = ct.reset_index().melt(id_vars=cat_feat, var_name=viz_target_select, value_name='Percentage')
                
                fig_stack = px.bar(ct_melt, x=cat_feat, y='Percentage', color=viz_target_select,
                                   title=f"Tỷ lệ {viz_target_select} theo nhóm {cat_feat}", text_auto='.1f',
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_stack, use_container_width=True)
            else: st.info("Không tìm thấy biến phân loại phù hợp.")

        # --- TAB 3: 3D ---
        with tab_3d:
            if len(numeric_vars) >= 3:
                c3d_1, c3d_2, c3d_3 = st.columns(3)
                with c3d_1: x_3d = st.selectbox("Trục X:", numeric_vars, index=0)
                with c3d_2: y_3d = st.selectbox("Trục Y:", numeric_vars, index=1 if len(numeric_vars)>1 else 0)
                with c3d_3: z_3d = st.selectbox("Trục Z:", numeric_vars, index=2 if len(numeric_vars)>2 else 0)
                
                try:
                    fig_3d = px.scatter_3d(df_clean, x=x_3d, y=y_3d, z=z_3d, color=viz_target_select,
                                           opacity=0.7, size_max=10, title=f"Không gian 3 chiều",
                                           color_discrete_sequence=px.colors.qualitative.Bold)
                    st.plotly_chart(fig_3d, use_container_width=True)
                except Exception as e: st.warning(f"Lỗi vẽ 3D: {e}")
            else: st.info("Cần ít nhất 3 biến số để vẽ 3D.")

        # --- 5. KIỂM ĐỊNH THỐNG KÊ (EXPANDER) ---
        st.markdown("---")
        with st.expander("🔬 Kiểm định Thống kê (T-test Report)", expanded=False):
            st.write("Kiểm tra ý nghĩa thống kê của sự khác biệt trung bình giữa 2 nhóm.")
            
            if len(df_clean[viz_target_select].unique()) == 2 and numeric_vars:
                stat_results = []
                groups = df_clean[viz_target_select].unique()
                
                for col in numeric_vars:
                    g1 = df_clean[df_clean[viz_target_select] == groups[0]][col]
                    g2 = df_clean[df_clean[viz_target_select] == groups[1]][col]
                    try:
                        t_stat, p_val = stats.ttest_ind(g1, g2, nan_policy='omit')
                        sig = "⭐⭐⭐ Có ý nghĩa" if p_val < 0.05 else "Không"
                        stat_results.append({"Biến số": col, "p-value": f"{p_val:.5f}", "Kết luận (p<0.05)": sig})
                    except: pass
                
                if stat_results:
                    st_df = pd.DataFrame(stat_results)
                    st.dataframe(st_df.style.applymap(lambda v: 'color: green; font-weight: bold' if 'Có ý nghĩa' in str(v) else '', subset=['Kết luận (p<0.05)']), use_container_width=True)
            else:
                st.info("Chỉ hỗ trợ bài toán phân loại nhị phân (2 nhóm) và cần có biến số.")

# ==================== TAB 3: CẤU HÌNH & HUẤN LUYỆN (FULL FEATURES) ====================
    with tab3:
        # Import thư viện nâng cao
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from sklearn.calibration import calibration_curve, CalibrationDisplay
        import time # Để tạo timestamp cho tên mô hình
        
        # Thư viện Imbalance nâng cao
        try:
            from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        except ImportError:
            st.error("Thiếu thư viện imblearn. Vui lòng cài đặt: pip install imbalanced-learn")
            st.stop()

        if 'df_main' in st.session_state:
            df_clean = st.session_state['df_main']
            
        # --- KHỞI TẠO LỊCH SỬ LƯU TRỮ ---
        if 'run_history' not in st.session_state:
            st.session_state['run_history'] = []
            
        # --- KHỞI TẠO KHO MODEL (QUAN TRỌNG CHO TAB 4) ---
        if 'trained_models' not in st.session_state:
            st.session_state['trained_models'] = {}

        st.header("🧠 Huấn luyện Mô hình & Tối ưu hóa (Advanced)")
        st.caption("Tích hợp tinh chỉnh tham số tự động (Hyperparameter Tuning) và so sánh hiệu năng.")

# --- PHẦN 0: LOAD MODEL & CẬP NHẬT LỊCH SỬ (ĐÃ SỬA LỖI UNBOUND LOCAL ERROR) ---
        with st.expander("📂 Tải lên mô hình đã lưu (.pkl)", expanded=False):
            uploaded_model = st.file_uploader("Chọn file model:", type=['pkl'])
            
            if uploaded_model is not None:
                try:
                    # 1. Load file
                    loaded_package = joblib.load(uploaded_model)
                    
                    if isinstance(loaded_package, dict) and 'model' in loaded_package:
                        # 2. Cập nhật vào Session State hiện tại
                        st.session_state.update(loaded_package)
                        st.session_state['is_trained'] = True
                        
                        # 3. TRÍCH XUẤT THÔNG TIN ĐỂ LƯU VÀO LỊCH SỬ
                        model_name = loaded_package.get('model_type', 'Loaded Model')
                        features_cnt = len(loaded_package.get('features', []))
                        
                        # Lấy Accuracy
                        acc = loaded_package.get('accuracy', 0.0)
                        recall = 0.0
                        f1 = 0.0
                        auc_val = 0.0
                        
                        # Tính lại chỉ số nếu có dữ liệu test cũ
                        if 'y_test' in loaded_package and 'y_prob' in loaded_package:
                            y_t = loaded_package['y_test']
                            y_p = loaded_package['y_prob']
                            y_p_class = (y_p >= 0.5).astype(int)
                            
                            acc = accuracy_score(y_t, y_p_class)
                            recall = recall_score(y_t, y_p_class, zero_division=0)
                            f1 = f1_score(y_t, y_p_class, zero_division=0)
                            try:
                                auc_val = roc_auc_score(y_t, y_p)
                            except: auc_val = 0.0

                        # 4. Tạo Log và thêm vào Run History
                        timestamp_str = pd.Timestamp.now().strftime('%H:%M:%S')
                        
                        # Xử lý tên và deduplication (xóa trùng)
                        model_name_clean = model_name.split('(')[0].strip()
                        
                        log_entry = {
                            "Run ID": f"{model_name} (Load) - {timestamp_str}",
                            "Model": model_name_clean,
                            "Accuracy": acc,
                            "Sensitivity (Recall)": recall,
                            "Precision": 0.0, # Mặc định nếu file cũ không có
                            "F1-Score": f1,
                            "F2-Score": f1,   # Tạm lấy F1 nếu file cũ không có F2
                            "AUC": auc_val,
                            "Features Count": features_cnt
                        }
                        
                        if 'run_history' not in st.session_state:
                            st.session_state['run_history'] = []
                        
                        # LOGIC QUAN TRỌNG: Xóa dòng cũ nếu cùng tên Model
                        st.session_state['run_history'] = [
                            entry for entry in st.session_state['run_history'] 
                            if entry['Model'] != model_name_clean
                        ]
                            
                        st.session_state['run_history'].append(log_entry)

                        st.success(f"✅ Đã tải & Cập nhật lịch sử: **{model_name_clean}**")
                    else:
                        st.error("⚠️ File không đúng định dạng (thiếu key 'model').")
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")

        st.markdown("---")

        # --- PHẦN 1: CẤU HÌNH ---
        col_setup1, col_setup2 = st.columns([1, 2])
        
        with col_setup1:
            st.subheader("1. Dữ liệu & Thuật toán")
            # Chọn Target
            priority_targets = ['HEART FAILURE', 'ACS', 'STEMI', 'TARGET', 'OUTPUT', 'DEATH_EVENT']
            def_idx = 0
            for t in priority_targets:
                if t in df_clean.columns:
                    def_idx = list(df_clean.columns).index(t)
                    break
            target_col = st.selectbox("Cột Mục tiêu (Target):", df_clean.columns, index=def_idx)
            
            # Chọn Model
            model_type = st.selectbox(
                "🔹 Chọn Thuật toán:", 
                ["Support Vector Machine (SVM)", "Random Forest (Rừng ngẫu nhiên)", "Logistic Regression (Hồi quy Logistic)", "K-Nearest Neighbors (KNN)","XGBoost (Gradient Boosting)"]
            )
            
            # Xử lý Feature Mặc định 
            all_features = [c for c in df_clean.columns if c != target_col]
            requested_defaults = ['EF', 'BNP', 'HTN', 'DM', 'CAD', 'CKD', 'AGE', 'HB', 'UREA', 'CREATININE', 'GLUCOSE', 'VALVULAR']
            default_feats = [f for f in requested_defaults if f in all_features]
            if not default_feats: default_feats = all_features[:10]

            # Logic AI Override
            if 'ai_suggested_features' in st.session_state and st.session_state.get('use_ai_features', False):
                valid_ai = [f for f in st.session_state['ai_suggested_features'] if f in all_features]
                if valid_ai: default_feats = valid_ai
            
            selected_features = st.multiselect("Chọn Input Features:", all_features, default=default_feats)
            
            c_ai1, c_ai2 = st.columns(2)
            with c_ai1:
                if st.button("✨ Dùng AI gợi ý"):
                    if 'ai_suggested_features' in st.session_state:
                        st.session_state['use_ai_features'] = True
                        st.rerun()
                    else: st.warning("Cần chạy Tab 1 trước.")
            with c_ai2:
                if st.session_state.get('use_ai_features', False):
                    if st.button("↺ Mặc định"):
                        st.session_state['use_ai_features'] = False
                        st.rerun()

        with col_setup2:
            st.subheader("2. Cấu hình Nâng cao")
            
            # Tab cấu hình: Thủ công vs Tự động
            mode_train = st.radio("Chế độ huấn luyện:", ["Thủ công (Manual)", "Tự động tối ưu (Auto-Tune)"], horizontal=True)

            if mode_train == "Thủ công (Manual)":
                if model_type == "Support Vector Machine (SVM)":
                    c1, c2 = st.columns(2)
                    kernel = c1.selectbox("Kernel", ["rbf", "linear"])
                    c_param = c2.number_input("C (Regularization)", 0.01, 100.0, 1.0)
                elif model_type == "Random Forest (Rừng ngẫu nhiên)":
                    c1, c2 = st.columns(2)
                    n_estimators = c1.slider("Số cây", 10, 200, 100)
                    max_depth = c2.slider("Độ sâu", 1, 20, 10)
                elif model_type == "K-Nearest Neighbors (KNN)":
                    n_neighbors = st.slider("K (Neighbors)", 1, 21, 5)
                elif model_type == "Logistic Regression (Hồi quy Logistic)":
                    c_param = st.number_input("C (Inv Reg)", 0.01, 100.0, 1.0)
                elif model_type == "XGBoost (Gradient Boosting)":
                    c1, c2, c3 = st.columns(3)
                    n_estimators = c1.slider("Số cây (Estimators)", 50, 500, 100)
                    max_depth = c2.slider("Độ sâu (Max Depth)", 1, 20, 6)
                    learning_rate = c3.number_input("Learning Rate", 0.01, 1.0, 0.3, 0.01)
            else:
                st.info(f"⚡ Hệ thống sẽ chạy **RandomizedSearchCV** để tìm tham số tối ưu nhất cho **{model_type}**.")
                n_iter_search = st.slider("Số lần thử nghiệm (n_iter)", 5, 50, 10, help="Càng cao càng chính xác nhưng càng lâu.")

            st.markdown("---")
            st.write("**Xử lý Mất cân bằng (Imbalance Handling):**")
            c_imb1, c_imb2 = st.columns(2)
            with c_imb1:
                imb_method = st.selectbox("Phương pháp:", ["Không", "SMOTE (Cơ bản)", "ADASYN (Thích ứng)", "BorderlineSMOTE"])
            with c_imb2:
                test_size = st.slider("Tỷ lệ Test (%)", 10, 40, 20) / 100

        # --- PHẦN 2: ACTION ---
        st.markdown("---")
        if st.button("🚀 HUẤN LUYỆN (TRAIN NOW)", type="primary"):
            if not selected_features:
                st.error("Vui lòng chọn đặc trưng đầu vào.")
            else:
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    # B1: Split Data
                    status.text("Đang chuẩn bị dữ liệu...")
                    X = df_clean[selected_features]
                    y = df_clean[target_col]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                    progress_bar.progress(10)

                    # B2: Imbalance Handling (Nâng cao)
                    status.text(f"Đang xử lý cân bằng dữ liệu bằng {imb_method}...")
                    if imb_method != "Không":
                        try:
                            if imb_method == "SMOTE (Cơ bản)": sampler = SMOTE(random_state=42)
                            elif imb_method == "ADASYN (Thích ứng)": sampler = ADASYN(random_state=42)
                            elif imb_method == "BorderlineSMOTE": sampler = BorderlineSMOTE(random_state=42)
                            
                            X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)
                        except Exception as err:
                            st.warning(f"Lỗi thuật toán {imb_method} (có thể do dữ liệu quá ít). Chuyển về SMOTE thường. Chi tiết: {err}")
                            sampler = SMOTE(random_state=42)
                            X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)
                    else:
                        X_train_bal, y_train_bal = X_train, y_train
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_bal)
                    X_test_scaled = scaler.transform(X_test)
                    progress_bar.progress(30)

                    # B3: Training Logic (Manual vs Auto-Tune)
                    final_model = None
                    best_params_found = "Manual Configuration"

                    if mode_train == "Thủ công (Manual)":
                        status.text(f"Đang huấn luyện {model_type} (Thủ công)...")
                        if model_type == "Support Vector Machine (SVM)":
                            final_model = SVC(kernel=kernel, C=c_param, probability=True, random_state=42)
                        elif model_type == "Random Forest (Rừng ngẫu nhiên)":
                            final_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                        elif model_type == "Logistic Regression (Hồi quy Logistic)":
                            final_model = LogisticRegression(C=c_param, random_state=42)
                        elif model_type == "K-Nearest Neighbors (KNN)":
                            final_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                        elif model_type == "XGBoost (Gradient Boosting)":
                            final_model = XGBClassifier(
                                n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                learning_rate=learning_rate, 
                                eval_metric='logloss', 
                                random_state=42
                            )
                        final_model.fit(X_train_scaled, y_train_bal)
                    
                    else: # Auto-Tune
                        status.text(f"Đang tối ưu Hyperparameters (Random Search)...")
                        param_dist = {}
                        base_estimator = None
                        
                        if model_type == "Support Vector Machine (SVM)":
                            base_estimator = SVC(probability=True, random_state=42)
                            param_dist = {
                                'C': [0.1, 1, 10, 100], 
                                'kernel': ['linear', 'rbf', 'poly'],
                                'gamma': ['scale', 'auto', 0.1, 0.01]
                            }
                        elif model_type == "Random Forest (Rừng ngẫu nhiên)":
                            base_estimator = RandomForestClassifier(random_state=42)
                            param_dist = {
                                'n_estimators': [50, 100, 200, 300],
                                'max_depth': [None, 10, 20, 30],
                                'min_samples_split': [2, 5, 10]
                            }
                        elif model_type == "Logistic Regression (Hồi quy Logistic)":
                            base_estimator = LogisticRegression(random_state=42)
                            param_dist = {'C': [0.01, 0.1, 1, 10, 100]}
                        elif model_type == "K-Nearest Neighbors (KNN)":
                            base_estimator = KNeighborsClassifier()
                            param_dist = {'n_neighbors': range(3, 20), 'weights': ['uniform', 'distance']}
                        elif model_type == "XGBoost (Gradient Boosting)":
                            base_estimator = XGBClassifier(eval_metric='logloss', random_state=42)
                            param_dist = {
                                'n_estimators': [50, 100, 200, 300],
                                'max_depth': [3, 5, 7, 10],
                                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                                'subsample': [0.6, 0.8, 1.0]
                            }
                        search = RandomizedSearchCV(base_estimator, param_distributions=param_dist, 
                                                    n_iter=n_iter_search, cv=3, scoring='recall', random_state=42, n_jobs=-1)
                        search.fit(X_train_scaled, y_train_bal)
                        final_model = search.best_estimator_
                        best_params_found = str(search.best_params_)

                    progress_bar.progress(80)

                    # B4: Predict & Evaluate
                    status.text("Đang tính toán Metrics & Vẽ biểu đồ...")
                    y_prob = final_model.predict_proba(X_test_scaled)[:, 1]
                    y_pred = final_model.predict(X_test_scaled)
                    
                    # Permutation Importance
                    perm = permutation_importance(final_model, X_test_scaled, y_test, n_repeats=5, random_state=42)
                    feat_imp = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': perm.importances_mean
                    }).sort_values(by='Importance', ascending=False)
                    
                    # --- [MỚI] TÍNH TOÁN METRICS ĐẦY ĐỦ ---
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0) # F2-Score
                    
                    auc_val = roc_auc_score(y_test, y_prob) if len(np.unique(y_test))>1 else 0
                    
                    timestamp_short = time.strftime("%H:%M:%S")
                    run_name_full = f"{model_type.split('(')[0].strip()} ({'Auto' if mode_train!='Thủ công (Manual)' else 'Manual'})"
                    model_name_clean = model_type.split('(')[0].strip()

                    # 1. Tạo Log Entry
                    log_entry = {
                        "Run ID": f"{run_name_full} - {timestamp_short}",
                        "Model": model_name_clean,
                        "Accuracy": acc,
                        "Sensitivity (Recall)": recall,
                        "Precision": precision,
                        "F1-Score": f1,
                        "F2-Score": f2, 
                        "AUC": auc_val,
                        "Features Count": len(selected_features)
                    }
                    
                    # 2. [QUAN TRỌNG] Xóa cũ - Đè mới (Deduplication)
                    if 'run_history' not in st.session_state:
                        st.session_state['run_history'] = []

                    # Giữ lại các model KHÁC tên với model đang chạy
                    st.session_state['run_history'] = [
                        entry for entry in st.session_state['run_history'] 
                        if entry['Model'] != model_name_clean
                    ]
                    
                    # Thêm kết quả mới nhất vào
                    st.session_state['run_history'].append(log_entry)
                    
                    # 2. Lưu vào Kho Model (SỬA: THÊM METRICS ĐỂ TAB 4 KHÔNG BỊ 0.0%)
                    model_key = f"{model_type} ({timestamp_short}) - Acc: {acc:.1%}"
                    st.session_state['trained_models'][model_key] = {
                        'model': final_model,
                        'scaler': scaler,
                        'features': selected_features,
                        'target_name': target_col,
                        'feature_importance': feat_imp,
                        'model_type': model_type,
                        'le_dict': st.session_state.get('le_dict', {}),
                        'run_time': timestamp_short,
                        # --- THÊM CÁC DÒNG NÀY ---
                        'accuracy': acc,
                        'f2_score': f2,
                        'recall': recall,
                        'precision': precision,
                        'f1_score': f1,
                        'auc': auc_val
                        # -------------------------
                    }
                    # 3. Lưu Session hiện tại
                    st.session_state['model'] = final_model
                    st.session_state['model_type'] = model_type
                    st.session_state['scaler'] = scaler
                    st.session_state['features'] = selected_features
                    st.session_state['target_name'] = target_col
                    st.session_state['feature_importance'] = feat_imp
                    st.session_state['X_test_scaled'] = X_test_scaled
                    st.session_state['y_test'] = y_test
                    st.session_state['y_prob'] = y_prob
                    st.session_state['best_params'] = best_params_found
                    st.session_state['is_trained'] = True
                    
                    progress_bar.progress(100)
                    status.success(f"✅ Huấn luyện hoàn tất! (Đã lưu vào bộ nhớ)")
                    # === ĐÂY LÀ PHẦN QUAN TRỌNG CẦN SỬA/THÊM ===
                    
                    # 1. Lưu vào Kho Model (Để Tab 4 dùng được)
                    timestamp_short = time.strftime("%H:%M:%S")
                    
                    # 2. LƯU CÁC BIẾN QUAN TRỌNG VÀO SESSION STATE (SỬA LỖI KEYERROR TẠI ĐÂY)
                    st.session_state['model'] = final_model
                    st.session_state['model_type'] = model_type
                    st.session_state['scaler'] = scaler
                    st.session_state['features'] = selected_features
                    st.session_state['target_name'] = target_col
                    
                    # ---> DÒNG CẦN THÊM ĐỂ HẾT LỖI <---
                    st.session_state['X_test_scaled'] = X_test_scaled  
                    st.session_state['y_test'] = y_test
                    # -------------------------------------
                    
                    st.session_state['is_trained'] = True
                    
                    progress_bar.progress(100)
                    status.success(f"✅ Huấn luyện hoàn tất! (Đã lưu vào bộ nhớ)")
                except Exception as e:
                    st.error(f"Lỗi Training: {e}")

        # --- PHẦN 3: KẾT QUẢ & ĐÁNH GIÁ CHUYÊN SÂU ---
        if st.session_state.get('is_trained', False):
            st.markdown("---")
            st.header(f"📊 Kết quả Đánh giá: {st.session_state.get('model_type')}")
            
            # Hiển thị Best Params nếu chạy Auto-tune
            if 'best_params' in st.session_state and "Manual" not in st.session_state['best_params']:
                st.success(f"🏆 **Tham số tối ưu tìm được:** {st.session_state['best_params']}")
            
            y_test = st.session_state['y_test']
            y_prob = st.session_state['y_prob']
            
            # 1. Metrics cơ bản
            c_res1, c_res2 = st.columns([1, 2])
            with c_res1:
                st.subheader("1. Medical Metrics (Chỉ số Y tế)")
                # Slider điều chỉnh ngưỡng
                threshold = st.slider("Ngưỡng quyết định (Threshold)", 0.0, 1.0, 0.5, 0.01)
                
                # Tính toán dự báo dựa trên ngưỡng mới
                y_pred_adj = (y_prob >= threshold).astype(int)
                
                # --- TÍNH TOÁN LẠI CÁC CHỈ SỐ ---
                acc = accuracy_score(y_test, y_pred_adj)
                sens = recall_score(y_test, y_pred_adj, zero_division=0)
                prec = precision_score(y_test, y_pred_adj, zero_division=0)
                f1 = f1_score(y_test, y_pred_adj, zero_division=0)
                f2 = fbeta_score(y_test, y_pred_adj, beta=2, zero_division=0) # Tính lại F2
                
                cm = confusion_matrix(y_test, y_pred_adj)
                spec = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
                auc_score = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
                
                # --- [LOGIC MỚI] CẬP NHẬT NGƯỢC LẠI VÀO LỊCH SỬ (REAL-TIME UPDATE) ---
                # Mục đích: Để Leaderboard và Radar Chart phía dưới thay đổi theo Slider
                if 'run_history' in st.session_state:
                    current_model_clean = st.session_state.get('model_type', '').split('(')[0].strip()
                    for entry in st.session_state['run_history']:
                        # Chỉ cập nhật dòng tương ứng với model đang chỉnh
                        if entry['Model'] == current_model_clean:
                            entry['Accuracy'] = acc
                            entry['Sensitivity (Recall)'] = sens
                            entry['Precision'] = prec
                            entry['F1-Score'] = f1
                            entry['F2-Score'] = f2
                            entry['Threshold'] = threshold # Lưu lại ngưỡng hiện tại
                
                # Hiển thị bảng chỉ số
                met_df = pd.DataFrame([
                    {"Chỉ số": "Sensitivity (Độ nhạy)", "Giá trị": sens},
                    {"Chỉ số": "Precision (Độ chính xác TP)", "Giá trị": prec},
                    {"Chỉ số": "Specificity (Độ đặc hiệu)", "Giá trị": spec},
                    {"Chỉ số": "Accuracy (Độ chính xác)", "Giá trị": acc},
                    {"Chỉ số": "F1-Score", "Giá trị": f1},
                    {"Chỉ số": "AUC (Diện tích dưới ROC)", "Giá trị": auc_score}
                ])
                
                st.dataframe(
                    met_df.style.format({"Giá trị": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True
                )
                
            with c_res2:
                st.subheader("2. Biểu đồ Phân tích")
                t1, t2, t3, t4 = st.tabs(["Confusion Matrix", "ROC Curve", "Calibration Plot", "Decision Curve Analysis"])
                
                with t1:
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                      labels=dict(x="Dự báo", y="Thực tế", color="Số lượng"),
                                      x=["Âm tính", "Dương tính"], y=["Âm tính", "Dương tính"])
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with t2:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {auc_score:.3f})",
                                      labels=dict(x="1 - Specificity (FPR)", y="Sensitivity (TPR)"))
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc, use_container_width=True)

                with t3:
                    # Calibration Plot
                    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Model'))
                    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfectly Calibrated', line=dict(dash='dash')))
                    fig_cal.update_layout(title="Calibration Plot (Độ tin cậy)", xaxis_title="Xác suất dự báo", yaxis_title="Tỷ lệ thực tế")
                    st.plotly_chart(fig_cal, use_container_width=True)
                    st.caption("Đường càng gần nét đứt, xác suất dự báo càng đáng tin cậy.")

                with t4:
                    # Decision Curve Analysis
                    thresh_list = np.linspace(0.01, 0.99, 100)
                    net_benefits = []
                    for thr in thresh_list:
                        y_pred_thr = (y_prob >= thr).astype(int)
                        tp = np.sum((y_test == 1) & (y_pred_thr == 1))
                        fp = np.sum((y_test == 0) & (y_pred_thr == 1))
                        n = len(y_test)
                        nb = (tp / n) - (fp / n) * (thr / (1 - thr))
                        net_benefits.append(nb)
                    
                    fig_dca = go.Figure()
                    fig_dca.add_trace(go.Scatter(x=thresh_list, y=net_benefits, mode='lines', name='Model Net Benefit'))
                    # Treat All
                    prevalence = np.sum(y_test) / len(y_test)
                    net_benefit_all = prevalence - (1 - prevalence) * (thresh_list / (1 - thresh_list))
                    fig_dca.add_trace(go.Scatter(x=thresh_list, y=net_benefit_all, mode='lines', name='Treat All', line=dict(dash='dot', color='gray')))
                    fig_dca.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode='lines', name='Treat None', line=dict(color='black')))
                    
                    fig_dca.update_layout(title="Decision Curve Analysis (Lợi ích lâm sàng)", xaxis_title="Ngưỡng xác suất", yaxis_title="Lợi ích ròng (Net Benefit)", yaxis_range=[-0.1, 0.5])
                    st.plotly_chart(fig_dca, use_container_width=True)
                    st.caption("DCA cho biết liệu sử dụng mô hình có đem lại lợi ích thực tế so với việc không làm gì hoặc chữa trị cho tất cả mọi người.")
            
            st.markdown("---")
            
            # --- KHỐI 1: KẾT QUẢ MÔ HÌNH HIỆN TẠI (Thêm chi tiết realtime) ---
            c_res1, c_res2 = st.columns([1, 2])
            with c_res1:
                # Đã hiển thị ở trên, có thể ẩn bớt để tránh lặp lại nếu muốn, 
                # nhưng giữ lại logic metric realtime nếu người dùng kéo slider threshold
                pass 

            # =========================================================
            # [GIAO DIỆN MỚI] LEADERBOARD: F2-SCORE + CUSTOM WEIGHTS
            # =========================================================
            st.markdown("---")
            st.header("⚔️ Bảng Xếp hạng & So sánh Đa chiều")
            
            if len(st.session_state['run_history']) > 0:
                history_list = st.session_state['run_history']
                
                # Fix lỗi key nếu dữ liệu cũ thiếu
                for h in history_list:
                    if 'F2-Score' not in h: h['F2-Score'] = 0
                    if 'Precision' not in h: h['Precision'] = 0

                # 1. CHỌN CHẾ ĐỘ XẾP HẠNG
                c_mode, c_blank = st.columns([2, 3])
                with c_mode:
                    sort_mode = st.radio(
                        "🎯 Tiêu chí ",
                        ["Tiêu chuẩn Y tế (F2-Score)", "Tùy chỉnh Trọng số (Custom)"],
                        help="""
                        **Giải thích chi tiết:**
                        
                        1. **Tiêu chuẩn Y tế (F2-Score):**
                           - Công thức: Ưu tiên Độ nhạy (Recall) gấp đôi Độ chính xác.
                           - Dùng cho: Tầm soát bệnh (thà báo nhầm còn hơn bỏ sót).
                        
                        2. **Độ ưu tiên Tùy chỉnh (Priority):**
                           - Tùy chỉnh giá trị các thông số ưu tiên
                        """,
                        horizontal=True
                    )

                # 2. CẤU HÌNH TRỌNG SỐ (CHỈ HIỆN KHI CHỌN CUSTOM)
                # Mặc định (F2)
                w_sens, w_prec, w_f1, w_acc = 0, 0, 0, 0
                is_custom_valid = True
                
                if sort_mode == "Tùy chỉnh Trọng số (Custom)":
                    with st.container():
                        st.info("👇 Nhập trọng số mong muốn (Tổng phải bằng 100%):")
                        w1, w2, w3, w4 = st.columns(4)
                        with w1:
                            w_sens = st.number_input("Recall %", 0, 100, 50, 5)
                        with w2:
                            w_prec = st.number_input("Precision %", 0, 100, 30, 5)
                        with w3:
                            w_f1 = st.number_input("F1-Score %", 0, 100, 20, 5)
                        with w4:
                            w_acc = st.number_input("Accuracy %", 0, 100, 0, 5)
                        
                        total = w_sens + w_prec + w_f1 + w_acc
                        if total != 100:
                            st.warning(f"⚠️ Tổng hiện tại: **{total}%**. Vui lòng điều chỉnh về **100%**.")
                            is_custom_valid = False
                        else:
                            st.success("✅ Tổng hợp lệ (100%). Đang áp dụng...")
                            
                # --- LƯU CẤU HÌNH XẾP HẠNG ĐỂ TAB 4 DÙNG ---
                st.session_state['ranking_config'] = {
                    'mode': sort_mode,
                    'weights': {
                        'recall': w_sens if 'w_sens' in locals() else 0,
                        'prec': w_prec if 'w_prec' in locals() else 0,
                        'f1': w_f1 if 'w_f1' in locals() else 0,
                        'acc': w_acc if 'w_acc' in locals() else 0
                    },
                    'is_valid': is_custom_valid
                }
                # ------------------------------------------------

                # 3. XỬ LÝ SẮP XẾP & LƯU CẤU HÌNH (SỬA: ĐỂ TAB 4 ĐỒNG BỘ)
                
                # Lưu cấu hình xếp hạng vào Session State
                st.session_state['ranking_config'] = {
                    'mode': sort_mode,
                    'weights': {
                        'recall': w_sens if 'w_sens' in locals() else 0,
                        'prec': w_prec if 'w_prec' in locals() else 0,
                        'f1': w_f1 if 'w_f1' in locals() else 0,
                        'acc': w_acc if 'w_acc' in locals() else 0
                    },
                    'is_valid': is_custom_valid if 'is_custom_valid' in locals() else True
                }

                # Logic sắp xếp hiển thị tại chỗ
                if sort_mode == "Tiêu chuẩn Y tế (F2-Score)":
                    sorted_history = sorted(history_list, key=lambda x: x.get('F2-Score', 0), reverse=True)
                else:
                    if is_custom_valid:
                        def calc_custom(row):
                            return (
                                row.get('Sensitivity (Recall)', 0) * (w_sens/100) +
                                row.get('Precision', 0) * (w_prec/100) +
                                row.get('F1-Score', 0) * (w_f1/100) +
                                row.get('Accuracy', 0) * (w_acc/100)
                            )
                        sorted_history = sorted(history_list, key=calc_custom, reverse=True)
                    else:
                        sorted_history = sorted(history_list, key=lambda x: x.get('F2-Score', 0), reverse=True)

                st.markdown("---")

                # 4. HIỂN THỊ (RADAR + LIST)
                chart_col, list_col = st.columns([4, 6])
                
                # --- PHẦN A: BIỂU ĐỒ RADAR ---
                with chart_col:
                    st.subheader("🕸️ So sánh Đa chiều")
                    if len(sorted_history) > 0:
                        categories = ['Accuracy', 'Sensitivity', 'Precision', 'F1-Score', 'F2-Score']
                        fig_radar = go.Figure()
                        colors = ['#00C853', '#2962FF', '#FF6D00'] 
                        
                        for idx, item in enumerate(sorted_history[:3]): 
                            values = [
                                item.get('Accuracy', 0), item.get('Sensitivity (Recall)', 0),
                                item.get('Precision', 0), item.get('F1-Score', 0), item.get('F2-Score', 0)
                            ]
                            values += values[:1]
                            fig_radar.add_trace(go.Scatterpolar(
                                r=values, theta=categories + categories[:1], fill='toself', 
                                name=item['Model'], line_color=colors[idx % 3], opacity=0.3
                            ))

                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=True, margin=dict(l=40, r=40, t=30, b=20), height=380,
                            legend=dict(orientation="h", y=-0.15),
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                # --- PHẦN B: DANH SÁCH CHI TIẾT ---
                with list_col:
                    st.subheader(f"🏆 Bảng xếp hạng: {sort_mode.split('(')[0]}")
                    
                    # Header
                    h1, h2, h3, h4, h5, h6 = st.columns([0.8, 2.5, 1.2, 1.2, 0.8, 0.8])
                    h1.markdown("<b>Hạng</b>", unsafe_allow_html=True)
                    h2.markdown("<b>Tên Mô hình</b>", unsafe_allow_html=True)
                    
                    # Đổi tên cột điểm số tùy chế độ
                    if sort_mode == "Tùy chỉnh Trọng số (Custom)" and is_custom_valid:
                        h3.markdown("<b>Custom Score</b>", unsafe_allow_html=True)
                    else:
                        h3.markdown("<b>F2-Score</b>", unsafe_allow_html=True)
                        
                    h4.markdown("<b>Recall</b>", unsafe_allow_html=True)
                    st.markdown("<div style='height: 1px; background-color: #ddd; margin-bottom: 10px;'></div>", unsafe_allow_html=True)

                    for i, row in enumerate(sorted_history):
                        rank_icon = f"#{i+1}"
                        bg_style = ""
                        border_color = "#eee"
                        if i == 0: 
                            rank_icon = "💎"
                            bg_style = "background-color: #e3f2fd;" 
                            border_color = "#2196F3"
                        elif i == 1: rank_icon = "🥇"
                        elif i == 2: rank_icon = "🥈"

                        with st.container():
                            st.markdown(f"""
                                <div style="{bg_style} border: 1px solid {border_color}; border-radius: 8px; padding: 8px 5px; margin-bottom: 8px;">
                                """, unsafe_allow_html=True)
                            
                            c1, c2, c3, c4, c5, c6 = st.columns([0.8, 2.5, 1.2, 1.2, 0.8, 0.8])
                            c1.write(f"**{rank_icon}**")
                            c2.write(f"**{row['Model']}**")
                            
                            # Hiển thị điểm số & Màu sắc
                            if sort_mode == "Tùy chỉnh Trọng số (Custom)" and is_custom_valid:
                                score_val = calc_custom(row)
                                c3.markdown(f"<span style='color:#D84315; font-weight:bold'>{score_val:.1%}</span>", unsafe_allow_html=True)
                            else:
                                c3.markdown(f"<span style='color:#2E7D32; font-weight:bold'>{row.get('F2-Score', 0):.1%}</span>", unsafe_allow_html=True)

                            c4.write(f"{row.get('Sensitivity (Recall)', 0):.1%}")
                            
                            with c5:
                                with st.popover("📖", use_container_width=True):
                                    thresh_val = row.get('Threshold', 0.5) # Mặc định 0.5 nếu chưa có
                                    st.write(f"**Ngưỡng (Threshold):** `{thresh_val:.2f}`") 
                                    st.write(f"**Recall:** `{row.get('Sensitivity (Recall)', 0):.1%}`")
                                    st.write(f"**Precision:** `{row.get('Precision', 0):.1%}`")
                                    st.write(f"**F1-Score:** `{row.get('F1-Score', 0):.1%}`")
                                    st.write(f"**Accuracy:** `{row.get('Accuracy', 0):.1%}`")
                                    
                                    if sort_mode == "Tùy chỉnh Trọng số (Custom)":
                                        st.markdown("---")
                                        st.caption(f"Trọng số áp dụng: Recall({w_sens}%) + Precision({w_prec}%) + F1({w_f1}%) + Acc({w_acc}%)")

                            with c6:
                                if st.button("🗑️", key=f"del_{row['Model']}_{i}"):
                                    st.session_state['run_history'] = [x for x in st.session_state['run_history'] if x['Model'] != row['Model']]
                                    st.rerun()
                            
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Chưa có dữ liệu lịch sử. Hãy thử huấn luyện mô hình!")
            # --- DOWNLOAD ---
            st.markdown("---")
            save_pkg = {
                'model': st.session_state['model'],
                'model_type': st.session_state.get('model_type', 'Unknown'),
                'scaler': st.session_state['scaler'],
                'features': st.session_state['features'],
                'target_name': st.session_state.get('target_name', 'Target'),
                'feature_importance': st.session_state.get('feature_importance', None),
                'le_dict': st.session_state.get('le_dict', {}),
                'y_test': st.session_state.get('y_test', None),
                'y_prob': st.session_state.get('y_prob', None),
                'best_params': st.session_state.get('best_params', None)
            }
            buf = io.BytesIO()
            joblib.dump(save_pkg, buf)
            st.download_button("💾 Tải xuống Model (.pkl)", data=buf.getvalue(), file_name=f"model_{st.session_state.get('model_type')}.pkl", mime="application/octet-stream")

    # ==================== TAB 4: QUẢN LÝ & TRIỂN KHAI (ADMIN) ====================
    with tab4:
        import os
        import datetime
        
        st.header("🚀 Trung tâm Kiểm soát Mô hình (Deployment Center)")
        st.caption("Quản lý mô hình đang hoạt động trên ứng dụng Chẩn đoán và triển khai mô hình mới.")

        # --- PHẦN A: TRẠNG THÁI HỆ THỐNG HIỆN TẠI (ĐỌC TỪ FILE) ---
        st.subheader("1. 🟢 Mô hình ĐANG HOẠT ĐỘNG (Active Model)")
        
        model_path = 'active_model.pkl'
        active_model_info = None

        if os.path.exists(model_path):
            try:
                active_pkg = joblib.load(model_path)
                
                # Lấy thời gian sửa đổi file
                mod_time = os.path.getmtime(model_path)
                dt_obj = datetime.datetime.fromtimestamp(mod_time)
                date_str = dt_obj.strftime('%d/%m/%Y %H:%M')
                
                # Lấy các chỉ số (Dùng .get để tránh lỗi nếu file cũ không có)
                active_model_info = {
                    'name': active_pkg.get('model_type', 'Unknown'),
                    'features': active_pkg.get('features', []),
                    'date': date_str,
                    # Metrics
                    'acc': active_pkg.get('accuracy', 0.0),
                    'f2': active_pkg.get('f2_score', 0.0),
                    'recall': active_pkg.get('recall', 0.0),
                    'precision': active_pkg.get('precision', 0.0),
                    'auc': active_pkg.get('auc', 0.0)
                }
                
                # Hiển thị thẻ thông tin Active
                with st.container():
                    st.success(f"✅ **Hệ thống đang chạy:** {active_model_info['name']}")
                    
                    # Hàng 1: Các chỉ số chính
                    c_act1, c_act2, c_act3, c_act4, c_act5 = st.columns(5)
                    c_act1.metric("Accuracy", f"{active_model_info['acc']:.1%}")
                    c_act2.metric("F2-Score (Y tế)", f"{active_model_info['f2']:.1%}")
                    c_act3.metric("Recall", f"{active_model_info['recall']:.1%}")
                    c_act4.metric("Precision", f"{active_model_info['precision']:.1%}")
                    c_act5.metric("AUC", f"{active_model_info['auc']:.3f}")

                    # Hàng 2: Thông tin phụ
                    c_inf1, c_inf2 = st.columns([3, 1])
                    c_inf1.caption(f"📅 Ngày kích hoạt: **{active_model_info['date']}** | 🔢 Số đặc trưng: **{len(active_model_info['features'])}**")
                    with c_inf2:
                        with st.expander("Xem đặc trưng"):
                            st.write(", ".join(active_model_info['features']))
                        
            except Exception as e:
                st.error(f"File model bị lỗi: {e}")
        else:
            st.warning("⚠️ Chưa có mô hình nào được kích hoạt! Hệ thống Bác sĩ đang ngừng hoạt động.")

        st.markdown("---")

        # --- PHẦN B: MÔ HÌNH ỨNG VIÊN (TỰ ĐỘNG LẤY TOP 1 TỪ TAB 3) ---
        st.subheader("2. 🔵 Mô hình ỨNG VIÊN (Candidate)")
        
        trained_models = st.session_state.get('trained_models', {})
        
        if not trained_models:
            st.info("👉 **Chưa có dữ liệu.** Vui lòng quay lại **Tab 3** để huấn luyện mô hình.")
        else:
            # 1. LẤY CẤU HÌNH TỪ TAB 3
            rank_config = st.session_state.get('ranking_config', {
                'mode': "Tiêu chuẩn Y tế (F2-Score)", 
                'weights': {}, 
                'is_valid': True
            })
            
            st.caption(f"⚡ Đang tự động chọn theo tiêu chuẩn: **{rank_config['mode']}**")

            # 2. HÀM TÍNH ĐIỂM (ĐỒNG BỘ VỚI TAB 3)
            def calculate_score(pkg):
                # Lấy chỉ số (Fallback về 0 nếu model cũ chưa có)
                f2 = pkg.get('f2_score', 0.0)
                recall = pkg.get('recall', 0.0)
                prec = pkg.get('precision', 0.0)
                f1 = pkg.get('f1_score', 0.0)
                acc = pkg.get('accuracy', 0.0)

                # Logic tính điểm Custom
                if rank_config['mode'] == "Tùy chỉnh Trọng số (Custom)" and rank_config['is_valid']:
                    w = rank_config['weights']
                    score = (
                        recall * (w['recall']/100) +
                        prec * (w['prec']/100) +
                        f1 * (w['f1']/100) +
                        acc * (w['acc']/100)
                    )
                    return score
                else:
                    return f2 # Mặc định F2

            # 3. SẮP XẾP VÀ LẤY TOP 1
            sorted_candidates = sorted(
                trained_models.values(), 
                key=calculate_score, 
                reverse=True
            )
            
            # Mặc định chọn thằng đứng đầu (Top 1)
            selected_pkg = sorted_candidates[0]
            
            # --- EXPANDER: CHỌN MÔ HÌNH KHÁC ---
            with st.expander("🔄 Chọn mô hình khác (từ Bảng xếp hạng)"):
                model_map = {}
                for m in sorted_candidates:
                    score = calculate_score(m)
                    # Tạo nhãn hiển thị
                    if rank_config['mode'] == "Tiêu chuẩn Y tế (F2-Score)":
                        score_txt = f"F2: {m.get('f2_score', 0):.1%}"
                    else:
                        score_txt = f"Score: {score:.1%}"
                        
                    label = f"{m['model_type']} ({score_txt}) - {m.get('run_time', '')}"
                    if m == sorted_candidates[0]:
                        label = "🏆 " + label # Đánh dấu Top 1
                    model_map[label] = m
                
                selected_label = st.selectbox(
                    "Danh sách mô hình (Đã sắp xếp):",
                    options=list(model_map.keys()),
                    index=0
                )
                
                if selected_label:
                    selected_pkg = model_map[selected_label]

            # 4. HIỂN THỊ CHI TIẾT
            cand_metrics = {
                'acc': selected_pkg.get('accuracy', 0.0),
                'f2': selected_pkg.get('f2_score', 0.0),
                'recall': selected_pkg.get('recall', 0.0),
                'precision': selected_pkg.get('precision', 0.0),
                'auc': selected_pkg.get('auc', 0.0)
            }
            
            def get_delta(key):
                if active_model_info and key in active_model_info:
                    return f"{cand_metrics[key] - active_model_info[key]:.1%}"
                return None

            with st.container():
                st.info(f"🔹 **Ứng viên được chọn:** {selected_pkg['model_type']}")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{cand_metrics['acc']:.1%}", get_delta('acc'))
                c2.metric("F2-Score", f"{cand_metrics['f2']:.1%}", get_delta('f2'))
                c3.metric("Recall", f"{cand_metrics['recall']:.1%}", get_delta('recall'))
                c4.metric("Precision", f"{cand_metrics['precision']:.1%}", get_delta('precision'))
                auc_v = cand_metrics['auc'] if cand_metrics['auc'] is not None else 0
                c5.metric("AUC", f"{auc_v:.3f}", get_delta('auc'))

                features_list = selected_pkg.get('features', [])
                st.write(f"**Đặc trưng sử dụng ({len(features_list)}):** {', '.join(features_list)}")

                # NÚT TRIỂN KHAI
                st.write("")
                if st.button("🚀 TRIỂN KHAI NGAY (Deploy)", type="primary", use_container_width=True):
                    try:
                        training_medians = {}
                        if 'df_main' in st.session_state:
                            training_medians = st.session_state['df_main'][features_list].median(numeric_only=True).to_dict()
                        
                        selected_pkg['training_medians'] = training_medians
                        # Lưu kèm cả metrics vào file active
                        joblib.dump(selected_pkg, 'active_model.pkl')
                        
                        st.balloons()
                        st.success(f"🎉 Đã kích hoạt: **{selected_pkg['model_type']}**")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
