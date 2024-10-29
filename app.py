import streamlit as st
import joblib
import pandas as pd
import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)

# Load models và scaler
@st.cache_resource
def load_models():
    try:
        lasso_model = joblib.load('model_linear.pkl')  # Bạn có thể thay thế bằng mô hình khác nếu cần
        scaler = joblib.load('scaler.pkl')
        feature_names = ['Hours_Studied', 'Previous_Scores', 'Extracurricular_Activities', 'Sleep_Hours', 'Sample_Question_Papers_Practiced']
        logging.info("Mô hình và scaler đã được tải thành công")
        return lasso_model, scaler, feature_names
    except Exception as e:
        logging.error(f"Lỗi khi tải mô hình hoặc scaler: {str(e)}")
        return None, None, None

lasso_model, scaler, feature_names = load_models()

if lasso_model is None or scaler is None:
    st.error("Không thể tải mô hình hoặc scaler. Vui lòng kiểm tra lại.")
else:
    # Nhập liệu từ người dùng
    st.title("Dự đoán hiệu suất học tập học sinh sinh viên")
    hours_studied = st.number_input("Số giờ học (0-9)", min_value=0, step=1)
    previous_scores = st.number_input("Điểm số trước đó (0-100)", min_value=0, step=1)
    extracurricular_activities = st.number_input("Hoạt động ngoại khóa (1: Có, 0: Không)", min_value=0, max_value=1, step=1)
    sleep_hours = st.number_input("Số giờ ngủ (0-9)", min_value=0, step=1)
    sample_question_papers_practiced = st.number_input("Số đề ôn tập đã làm (0-9)", min_value=0, step=1)

    # Khi nhấn "Dự đoán"
    if st.button("Dự đoán"):
        input_data = {
            'Hours_Studied': hours_studied,
            'Previous_Scores': previous_scores,
            'Extracurricular_Activities': extracurricular_activities,
            'Sleep_Hours': sleep_hours,
            'Sample_Question_Papers_Practiced': sample_question_papers_practiced
        }
        input_df = pd.DataFrame([input_data])

        # Kiểm tra tên cột và thứ tự
        if list(input_df.columns) != feature_names:
            st.error("Tên hoặc thứ tự các cột không khớp với dữ liệu huấn luyện ban đầu.")
            logging.error(f"Tên cột hiện tại: {list(input_df.columns)}")
            logging.error(f"Tên cột cần thiết: {feature_names}")
        else:
            try:
                # Chuẩn hóa dữ liệu
                input_scaled = scaler.transform(input_df)

                # Dự đoán với mô hình
                lasso_pred = lasso_model.predict(input_scaled)
                st.subheader("Kết quả dự đoán:")
                st.write(f"Dự đoán: {lasso_pred[0]}")
                logging.info(f"Kết quả dự đoán: {lasso_pred}")

            except Exception as e:
                logging.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
                st.error(f"Đã xảy ra lỗi: {str(e)}")
