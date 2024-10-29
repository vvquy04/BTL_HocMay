import streamlit as st
import joblib
import numpy as np
import pandas as pd
import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)

# Load các mô hình đã lưu và scaler
@st.cache_resource
def load_models():
    try:
        linear_model = joblib.load('model_linear.pkl')
        lasso_model = joblib.load('model_lasso.pkl')
        mlp_model = joblib.load('model_mlp.pkl')
        stacking_model = joblib.load('model_stack.pkl')
        scaler = joblib.load('scaler.pkl')
        logging.info("Các mô hình và scaler đã được tải thành công")
        return {'Linear': linear_model, 'Lasso': lasso_model, 'MLP': mlp_model, 'Stacking': stacking_model}, scaler
    except Exception as e:
        logging.error(f"Lỗi khi tải mô hình hoặc scaler: {str(e)}")
        return None, None

models, scaler = load_models()

# Kiểm tra xem mô hình có được tải thành công không
if models is None or scaler is None:
    st.error("Không thể tải các mô hình hoặc scaler. Vui lòng kiểm tra lại.")
else:
    # Hiển thị tiêu đề
    st.title("Dự đoán hiệu suất học tập học sinh sinh viên")

    # Chọn mô hình dự đoán
    st.subheader("Chọn mô hình dự đoán:")
    model_choice = st.selectbox("Mô hình", options=["Linear", "Lasso", "MLP", "Stacking"])

    # Nhập liệu từ người dùng
    st.subheader("Nhập thông tin sinh viên:")
    hours_studied = st.number_input("Số giờ học (0-9)", min_value=0, step=1)
    previous_scores = st.number_input("Điểm số trước đó (0-100)", min_value=0, step=1)
    extracurricular_activities = st.number_input("Hoạt động ngoại khóa (1: Có, 0: Không)", min_value=0, max_value=1, step=1)
    sleep_hours = st.number_input("Số giờ ngủ (0-9)", min_value=0, step=1)
    sample_question_papers_practiced = st.number_input("Số đề ôn tập đã làm (0-9)", min_value=0, step=1)

    # Khi người dùng nhấn vào nút "Dự đoán"
    if st.button("Dự đoán"):
        try:
            # Lấy dữ liệu đầu vào
            input_data = {
                'Hours_Studied': hours_studied,
                'Previous_Scores': previous_scores,
                'Extracurricular_Activities': extracurricular_activities,
                'Sleep_Hours': sleep_hours,
                'Sample_Question_Papers_Practiced': sample_question_papers_practiced
            }
            logging.info(f"Dữ liệu đầu vào: {input_data}")

            # Chuyển đổi dữ liệu thành DataFrame để sử dụng
            input_df = pd.DataFrame([input_data])

            # Chuẩn hóa dữ liệu
            input_scaled = scaler.transform(input_df)
            logging.info(f"Dữ liệu sau khi chuẩn hóa: {input_scaled}")

            # Dự đoán với mô hình được chọn
            model = models[model_choice]
            prediction = model.predict(input_scaled)
            logging.info(f"Kết quả dự đoán với mô hình {model_choice}: {prediction}")

            # Hiển thị kết quả
            st.subheader("Kết quả dự đoán:")
            st.write(f"Mô hình {model_choice} dự đoán: {prediction[0]}")

        except Exception as e:
            logging.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
            st.error(f"Đã xảy ra lỗi: {str(e)}")
