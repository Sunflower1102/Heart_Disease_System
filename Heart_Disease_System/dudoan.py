import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Thư viện giải thích AI
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    
# =========================================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN
# =========================================================
st.set_page_config(
    page_title="Doctor Assistant - Heart Diagnosis",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
    <style>
    .main-header { 
        font-size: 28px; color: #0D47A1; font-weight: bold; 
        text-align: center; margin-bottom: 20px; border-bottom: 2px solid #E3F2FD; padding-bottom: 10px;
    }
    .stButton>button { height: 3em; font-weight: bold; border-radius: 8px; }
    .metric-card {
        background-color: #f8f9fa; border: 1px solid #ddd;
        padding: 15px; border-radius: 8px; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# 2. HÀM LOAD MODEL & UTILS
# =========================================================
@st.cache_resource
def load_deployment_package():
    model_path = 'active_model.pkl'
    if not os.path.exists(model_path):
        return None
    try:
        package = joblib.load(model_path)
        return package
    except Exception as e:
        st.error(f"Lỗi đọc file model: {e}")
        return None

# =========================================================
# 3. HÀM HIỂN THỊ KẾT QUẢ 
# =========================================================
def show_prediction_result(inputs, pkg, missing_cols=[]):
    model = pkg['model']
    scaler = pkg['scaler']
    features = pkg['features']
    target_name = pkg['target_name']
    
    # --- [FIX 1] TỰ ĐỘNG KHÔI PHỤC FEATURE IMPORTANCE NẾU THIẾU ---
    imp_df = pkg.get('feature_importance', None)
    
    if imp_df is None:
        try:
            # Nếu model là Random Forest / Decision Tree
            if hasattr(model, 'feature_importances_'):
                imp_vals = model.feature_importances_
                imp_df = pd.DataFrame({'Feature': features, 'Importance': imp_vals})
            # Nếu model là Linear (Logistic, SVM Linear)
            elif hasattr(model, 'coef_'):
                imp_vals = np.abs(model.coef_[0])
                imp_df = pd.DataFrame({'Feature': features, 'Importance': imp_vals})
        except:
            imp_df = None # Không thể khôi phục (VD: KNN, SVM RBF)

    ref_values = pkg.get('training_medians', {}) 

    # 1. Chuẩn bị dữ liệu
    input_df = pd.DataFrame([inputs], columns=features)
    
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
        return

    # 2. Dự đoán
    try:
        prob = model.predict_proba(input_scaled)[0]
        risk_score = prob[1] * 100
    except:
        pred = model.predict(input_scaled)[0]
        risk_score = 100 if pred == 1 else 0
        prob = [1-pred, pred]

    # ================= GIAO DIỆN KẾT QUẢ =================
    st.markdown("---")
    st.subheader(f"📊 Kết quả Phân tích Lâm sàng")
    
    if missing_cols:
        st.warning(f"⚠️ **Lưu ý:** Có {len(missing_cols)} chỉ số bị thiếu/lỗi ({', '.join(missing_cols)}). Hệ thống đã tự động điền giá trị trung bình.")

    col_res1, col_res2 = st.columns([1.5, 1])

    with col_res1:
        st.markdown("#### Kết luận sơ bộ:")
        if risk_score > 50:
            st.error(f"### 🔴 DƯƠNG TÍNH (Nguy cơ cao - {risk_score:.1f}%)")
            st.write(f"Mô hình dự báo bệnh nhân có nguy cơ mắc **{target_name}**.")
        else:
            st.success(f"### 🟢 ÂM TÍNH (An toàn - {risk_score:.1f}%)")
            st.write(f"Các chỉ số hiện tại cho thấy bệnh nhân an toàn với **{target_name}**.")
            
        st.write(f"**Độ tin cậy của dự báo:** {max(prob)*100:.1f}%")
        
        st.info("💡 **Khuyến nghị:**")
        if risk_score > 80:
            st.markdown("- 🚨 **Hành động gấp:** Cần can thiệp y tế hoặc xét nghiệm chuyên sâu ngay.")
        elif risk_score > 50:
            st.markdown("- ⚠️ **Theo dõi:** Kiểm tra lại các chỉ số bất thường sau 2 tuần.")
        else:
            st.markdown("- ✅ **Duy trì:** Lối sống lành mạnh và tái khám định kỳ.")

    with col_res2:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            title = {'text': "Thang đo Rủi ro (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#D32F2F" if risk_score > 50 else "#2E7D32"},
                'steps': [
                    {'range': [0, 50], 'color': "#E8F5E9"}, 
                    {'range': [50, 75], 'color': "#FFF3E0"},
                    {'range': [75, 100], 'color': "#FFEBEE"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ================= GIẢI THÍCH (COMPARISON & IMPORTANCE) =================
    st.markdown("---")
    st.subheader("🔍 Giải thích: Tại sao có kết quả này?")
    
    if imp_df is not None:
        st.caption("So sánh các chỉ số quan trọng của bệnh nhân với mức 'Điển hình' (Median) của dữ liệu huấn luyện.")
        top_feats_list = imp_df.sort_values(by='Importance', ascending=False)['Feature'].head(10).tolist()
        available_feats = [f for f in top_feats_list if f in inputs]
        
        explanation_data = []
        for feat in available_feats:
            user_val = inputs[feat]
            ref_val = ref_values.get(feat, 0)
            
            eval_text = "N/A"
            if isinstance(user_val, (int, float)) and isinstance(ref_val, (int, float)) and ref_val != 0:
                diff_pct = ((user_val - ref_val) / ref_val) * 100
                if abs(diff_pct) > 10:
                    status = "Cao" if diff_pct > 0 else "Thấp"
                    icon = "🔺" if diff_pct > 0 else "🔻"
                    eval_text = f"{icon} {status} ({abs(diff_pct):.0f}%)"
                else:
                    eval_text = "✅ Bình thường"
            
            imp_score = imp_df[imp_df['Feature'] == feat]['Importance'].values[0] if not imp_df[imp_df['Feature'] == feat].empty else 0

            explanation_data.append({
                "Chỉ số": feat,
                "Độ quan trọng": imp_score,
                "Giá trị Bệnh nhân": f"{user_val:.2f}" if isinstance(user_val, float) else user_val,
                "Giá trị Điển hình": f"{ref_val:.2f}" if isinstance(ref_val, float) else ref_val,
                "Đánh giá": eval_text
            })
            
        res_df = pd.DataFrame(explanation_data)
        
        c_exp1, c_exp2 = st.columns([1.5, 1])
        with c_exp1:
            st.markdown("##### 📋 Bảng đối chiếu")
            st.dataframe(res_df.drop(columns=['Độ quan trọng']), use_container_width=True, hide_index=True)
            
        with c_exp2:
            st.markdown("##### 📊 Mức độ đóng góp")
            fig_exp = px.bar(
                res_df.sort_values(by="Độ quan trọng", ascending=True), 
                x='Độ quan trọng', y='Chỉ số', orientation='h',
                color='Độ quan trọng', color_continuous_scale='Reds'
            )
            fig_exp.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_exp, use_container_width=True)
    else:
        st.info("⚠️ Model hiện tại (VD: SVM RBF hoặc KNN) không hỗ trợ trích xuất độ quan trọng trực tiếp. Vui lòng xem biểu đồ SHAP bên dưới.")

    # ================= SHAP ANALYSIS =================
    st.markdown("---")
    st.subheader("🧬 Phân tích sâu (SHAP Values)")
    st.caption("Biểu đồ thác nước (Waterfall) giải thích chi tiết hướng tác động của từng biến số.")

    if HAS_SHAP and st.checkbox("👉 Hiển thị biểu đồ SHAP (Có thể tốn vài giây)"):
        with st.spinner("Đang tính toán SHAP..."):
            try:
                if ref_values:
                    medians_arr = np.array([pkg['training_medians'][f] for f in features])
                    background = np.tile(medians_arr, (5, 1))
                    noise = np.random.normal(0, 0.01, background.shape)
                    background = background + noise
                    background_scaled = scaler.transform(pd.DataFrame(background, columns=features))
                else:
                    st.warning("Thiếu dữ liệu tham chiếu để chạy SHAP.")
                    return

                if "RandomForest" in str(type(model)):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_scaled)
                    shap_val_display = shap_values[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    explainer = shap.KernelExplainer(model.predict_proba, background_scaled)
                    shap_values = explainer.shap_values(input_scaled)
                    shap_val_display = shap_values[1][0]
                    base_val = explainer.expected_value[1]

                fig_shap, ax = plt.subplots(figsize=(10, 6))
                exp_obj = shap.Explanation(
                    values=shap_val_display,
                    base_values=base_val,
                    data=input_scaled[0],
                    feature_names=features
                )
                shap.plots.waterfall(exp_obj, show=False)
                st.pyplot(fig_shap)
                
            except Exception as e:
                st.warning(f"Không thể tạo SHAP plot: {e}")

# =========================================================
# 4. CHƯƠNG TRÌNH CHÍNH (MAIN APP)
# =========================================================
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.sidebar.title("Doctor Assistant")
    
    pkg = load_deployment_package()
    
    if pkg is None:
        st.error("⚠️ **CHƯA KÍCH HOẠT HỆ THỐNG!**")
        st.warning("Vui lòng chạy `admin_app.py` -> Tab 4 -> Kích hoạt một mô hình.")
        if st.button("🔄 Tải lại trang"): st.rerun()
        return

    st.sidebar.markdown("---")
    st.sidebar.success(f"✅ Model: **{pkg.get('model_type', 'AI Model')}**")
    st.sidebar.info(f"Độ chính xác: **{pkg.get('accuracy', 0):.1%}**")
    st.sidebar.caption(f"Yêu cầu {len(pkg['features'])} chỉ số đầu vào.")

    st.title("🩺 Hệ thống Hỗ trợ Chẩn đoán Lâm sàng")
    st.markdown(f"**Mục tiêu dự đoán:** {pkg.get('target_name', 'Tim mạch')}")
    
    tab_manual, tab_batch = st.tabs(["✍️ Nhập liệu Thủ công (Khám lẻ)", "📂 Nhập liệu File (Hàng loạt)"])
    
    features = pkg['features']
    medians = pkg.get('training_medians', {})
    le_dict = pkg.get('le_dict', {})

    # --- TAB 1: NHẬP TAY ---
    with tab_manual:
        st.caption("Nhập các thông số lâm sàng của bệnh nhân.")
        with st.form("manual_form"):
            inputs = {}
            cols = st.columns(3)
            for i, feat in enumerate(features):
                with cols[i % 3]:
                    def_val = float(medians.get(feat, 0.0))
                    inputs[feat] = st.number_input(f"**{feat}**", value=def_val)
            
            st.markdown("---")
            submit_btn = st.form_submit_button("🏥 PHÂN TÍCH NGAY", type="primary")
            
        if submit_btn:
            show_prediction_result(inputs, pkg)

    # --- TAB 2: UPLOAD FILE ---
    with tab_batch:
        st.info(f"💡 File CSV cần chứa các cột: **{', '.join(features)}**")
        uploaded_file = st.file_uploader("Tải lên hồ sơ bệnh án (.csv):", type=['csv'])
        
        if uploaded_file:
            try:
                df_batch = pd.read_csv(uploaded_file)
                
                missing_required = [c for c in features if c not in df_batch.columns]
                if missing_required:
                    st.error(f"❌ File thiếu các cột: {', '.join(missing_required)}")
                else:
                    st.success(f"✅ Đã tải {len(df_batch)} hồ sơ. Đang xử lý...")
                    
                    # 1. XỬ LÝ DỮ LIỆU (FIX LỖI STRING)
                    X_batch = df_batch[features].copy()
                    
                    for col in X_batch.columns:
                        if X_batch[col].dtype == 'object':
                            if col in le_dict:
                                le = le_dict[col]
                                known_labels = set(le.classes_)
                                X_batch[col] = X_batch[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in known_labels else np.nan)
                            else:
                                X_batch[col] = pd.to_numeric(X_batch[col], errors='coerce')

                    # Track Missing
                    missing_tracker = X_batch.apply(lambda x: x.index[x.isnull()].tolist(), axis=1)
                    
                    # Impute
                    if medians:
                        X_batch_filled = X_batch.fillna(medians)
                    else:
                        X_batch_filled = X_batch.fillna(0)
                        
                    # Predict Batch
                    scaler = pkg['scaler']
                    model = pkg['model']
                    X_batch_scaled = scaler.transform(X_batch_filled)
                    probs = model.predict_proba(X_batch_scaled)[:, 1]
                    
                    df_display = df_batch.copy()
                    df_display['Nguy cơ (%)'] = (probs * 100).round(2)
                    df_display['Missing_Cols'] = missing_tracker
                    
                    id_col = next((c for c in df_display.columns if c.lower() in ['id', 'patient_id', 'name', 'ten']), None)
                    
                    df_display['Label_View'] = df_display.apply(
                        lambda x: f"{x[id_col] if id_col else 'Hồ sơ ' + str(x.name)} - Nguy cơ: {x['Nguy cơ (%)']}%" +
                                  (f" ⚠️(Lỗi/Thiếu số liệu)" if len(x['Missing_Cols']) > 0 else ""),
                        axis=1
                    )
                    
                    st.dataframe(df_display[['Nguy cơ (%)'] + features].head(10))
                    
                    st.markdown("---")
                    st.subheader("🔍 Xem chi tiết từng bệnh nhân")
                    
                    selected_patient_label = st.selectbox(
                        "Chọn hồ sơ để phân tích sâu:", 
                        df_display['Label_View'].tolist()
                    )
                    
                    if selected_patient_label:
                        row_idx = df_display[df_display['Label_View'] == selected_patient_label].index[0]
                        selected_row_filled = X_batch_filled.iloc[row_idx].to_dict()
                        missing_cols_row = df_display.iloc[row_idx]['Missing_Cols']
                        
                        show_prediction_result(selected_row_filled, pkg, missing_cols_row)

            except Exception as e:
                st.error(f"Lỗi xử lý file CSV: {e}")

if __name__ == "__main__":
    main()
