
# app.py
import streamlit as st
import requests

#  CẤU HÌNH 
BACKEND_URL = "http://127.0.0.1:8000"

# HÀM HỖ TRỢ (HELPER FUNCTIONS) 
def call_backend_api(endpoint, json_data=None, files=None):
    """
    Hàm chung để gọi API tới Backend FastAPI.
    Hỗ trợ cả gửi JSON data và File upload.
    """
    try:
        if files:
            # Nếu có file.  
            response = requests.post(f"{BACKEND_URL}/{endpoint}", files=files)
        else:
            response = requests.post(f"{BACKEND_URL}/{endpoint}", json=json_data)
            
        response.raise_for_status() 
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(" Lỗi: Không thể kết nối đến server Backend. Hãy đảm bảo FastAPI đang chạy.")
        return None
    except requests.exceptions.HTTPError:
        error_msg = response.json().get('error', 'Lỗi không xác định từ server')
        st.error(f" Lỗi API: {error_msg}")
        return None
    except Exception as e:
        st.error(f" Lỗi không xác định: {e}")
        return None

#  GIAO DIỆN (UI SETUP) 
st.set_page_config(page_title="AI Chatbot & Summarizer", page_icon="🤖", layout="centered")

# CSS 
st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    h1 {
        color: #333;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stAlert {
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🤖 Chatbot Tìm kiếm & Tóm tắt Văn bản</h1>", unsafe_allow_html=True)

#  PHẦN 1: AGENT TÌM KIẾM (GOOGLE SCHOLAR + GEMINI)

agent_query = st.text_input("Nhập câu hỏi nghiên cứu:", placeholder="Ví dụ: So sánh hiệu suất Llama 3 và GPT-4...", key="agent_query")

if st.button("🔍  Tìm kiếm", key="btn_agent_search", type="primary"):
    if agent_query.strip():
        with st.spinner("Agent đang tìm kiếm, đọc tài liệu và tổng hợp..."):
            response_data = call_backend_api("agent_search", json_data={"query": agent_query})
            
            if response_data:
                final_response = response_data.get("result", "")
                st.subheader(" Kết Quả Tổng Hợp ")
                st.markdown(final_response)
    else:
        st.warning("Vui lòng nhập câu hỏi ")

st.markdown("---")
#  PHẦN 2: XỬ LÝ VĂN BẢN (DỊCH & TÓM TẮT)
# Layout Nhập liệu 
col_text, col_file = st.columns([2, 1])
with col_text:
    st.caption(" **Nhập văn bản trực tiếp:**")
    typed_text = st.text_area("", height=250, placeholder="Dán bài báo hoặc đoạn văn cần xử lý vào đây...", 
                              key="typed_text_t5", label_visibility="collapsed")
    st.checkbox("Hiển thị nội dung gốc/dịch đầy đủ", key="show_full")      

with col_file:
    st.caption("**Hoặc tải file (PDF/DOCX):**")
    uploaded_file = st.file_uploader("", type=["pdf", "docx"], accept_multiple_files=False, 
                                     key="uploaded_file_t5", label_visibility="collapsed")
    
    st.caption("---")
    
    domain = st.selectbox("Lĩnh vực:", ["Công nghệ", "Khoa học", "Y tế", "Kinh tế", "Xu hướng", "Xã hội"], index=0, key="t5_domain")
    
    # Các nút chức năng
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        translate_btn = st.button(" Dịch", key="btn_translate")
    with btn_col2:
        trans_sum_btn = st.button(" Tóm tắt", key="btn_trans_sum")


#  Logic Xử lý Đầu vào 
input_text = ""

# Ưu tiên văn bản gõ, sau đó đến file upload
if typed_text and typed_text.strip():
    input_text = typed_text.strip()
elif uploaded_file:
    with st.spinner("Đang trích xuất nội dung file..."):
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        # Gọi API trích xuất file
        resp = call_backend_api("extract_text", files=files)
        if resp:
            input_text = resp.get("result", "")


#  Logic Nút Dịch 
if translate_btn:
    if not input_text:
        st.warning(" Vui lòng nhập văn bản hoặc tải file lên trước.")
    else:
        # 1. Kiểm tra ngôn ngữ
        lang_resp = call_backend_api("detect_language", json_data={"text": input_text})
        
        if lang_resp and lang_resp.get("is_vietnamese"):
            st.success("Văn bản gốc đã là Tiếng Việt.")
            st.subheader(" Nội dung Gốc")
            st.write(input_text)
        else:
            # 2. Gọi API Dịch
            with st.spinner("Đang dịch sang Tiếng Việt..."):
                trans_resp = call_backend_api("translate", json_data={"text": input_text})
                
                if trans_resp:
                    translated_text = trans_resp.get("result", "")
                    st.subheader("🌐 Nội dung (Đã dịch)")
                    
                    if st.session_state.get("show_full"):
                        st.write(translated_text)
                    else:
                        # Hiển thị rút gọn nếu dài
                        preview_len = 1500
                        st.write(translated_text[:preview_len] + ("..." if len(translated_text) > preview_len else ""))


#  Logic Nút Tóm tắt 
if trans_sum_btn:
    if not input_text:
        st.warning(" Vui lòng nhập văn bản hoặc tải file lên trước.")
    else:
        with st.spinner("Đang xử lý (Dịch & Tóm tắt)..."):
            text_to_summary = input_text
            
            # 1. Kiểm tra ngôn ngữ
            lang_resp = call_backend_api("detect_language", json_data={"text": input_text})
            
            # 2. Tự động dịch 
            if lang_resp and not lang_resp.get("is_vietnamese"):
                trans_resp = call_backend_api("translate", json_data={"text": input_text})
                if trans_resp:
                    text_to_summary = trans_resp.get("result", "")
                else:
                    text_to_summary = "" 
            
            # 3. Gọi API Tóm tắt
            if text_to_summary:
                sum_resp = call_backend_api("summarize", json_data={"text": text_to_summary, "domain": domain})
                
                if sum_resp:
                    summary_text = sum_resp.get("result", "")
                    st.subheader(" Tóm tắt ")
                    st.success(summary_text)