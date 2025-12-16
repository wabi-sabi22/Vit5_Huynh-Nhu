# search_agent.py
import os
import time
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Load environment variables (API Key)
load_dotenv()

# --- Cấu hình Client ---
try:
    # Google AI Client 
    gemini_client = genai.Client()
except Exception as e:
    print(f"Lỗi khởi tạo Gemini Client: {e}. Vui lòng kiểm tra GOOGLE_API_KEY.")
    gemini_client = None

# Serper API Key 
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
    print("Cảnh báo: Không tìm thấy SERPER_API_KEY. Chức năng Agent sẽ không hoạt động.")


# Hàm Tìm kiếm Serper 
def serper_search(query: str, num_results: int = 5) -> dict:
    """
    Tìm kiếm học thuật bằng Serper API trực tiếp (dùng requests).
    """
    if not SERPER_API_KEY:
        return {"error": "SERPER_API_KEY không được cấu hình"}
    
    url = "https://google.serper.dev/scholar"
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    payload = {
        'q': query,
        'num': num_results,
        'gl': 'vn',
        'hl': 'vi'
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gọi Serper API: {e}")
        return {"error": str(e)}


# Hàm Dịch Văn bản 
def translate_text(text: str, target_language: str = 'Vietnamese', max_retries: int = 3) -> str:
    """
    Dịch văn bản bằng Gemini API, có hỗ trợ Thử lại Tăng dần (Exponential Backoff)
    cho các lỗi tạm thời như 503 UNAVAILABLE.
    """
    if gemini_client is None:
        return "Lỗi: Gemini Client không khả dụng. Không thể dịch."

    system_instruction = (
        f"Bạn là một dịch giả chuyên nghiệp. Hãy dịch văn bản sau sang ngôn ngữ {target_language}. "
        f"Đảm bảo dịch chính xác, giữ nguyên ngữ cảnh và định dạng (ví dụ: các đoạn xuống dòng)."
    )
    
    # Chia nhỏ văn bản 
    CHUNK_SIZE = 7000 
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    translated_texts = []
    
    if not chunks:
        return ""
    
    for i, chunk in enumerate(chunks):
        
        for attempt in range(max_retries):
            try:
                # 1. Gọi API 
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=chunk,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                )

                # 2. Xử lý kết quả và thoát vòng lặp thử lại
                translated_texts.append(response.text)
                break
            except APIError as e:
                # Bắt các lỗi tạm thời (503, 429)
                if e.response.status_code in [503, 429] and attempt < max_retries - 1:
                    wait_time = 2 ** attempt # 1s, 2s, 4s
                    print(f"Lỗi API {e.response.status_code} (Quá tải/Throttled). Đang thử lại sau {wait_time} giây... (Lần {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Lỗi API nghiêm trọng khi dịch đoạn {i + 1}: {e}")
                    return f"Lỗi dịch API: {e}. Vui lòng kiểm tra GOOGLE_API_KEY hoặc thử lại sau."
            
            except Exception as e:
                print(f"Lỗi không xác định khi dịch đoạn {i + 1}: {e}")
                return f"Lỗi không xác định: {e}"

    return ' '.join(translated_texts)


# Hàm RAG Agent 
def rag_agent(query: str, max_retries: int = 3) -> str:
    """
    Sử dụng Serper để tìm kiếm học thuật và Gemini để tổng hợp thông tin, 
    cung cấp kết quả chi tiết kèm trích dẫn nguồn.
    Có cơ chế retry cho lỗi 503 (overloaded).
    """
    if gemini_client is None:
        return "Lỗi: Gemini Client không khả dụng. Vui lòng kiểm tra GOOGLE_API_KEY."
    if not SERPER_API_KEY:
        return "Lỗi: SERPER_API_KEY không được cấu hình. Agent không thể tìm kiếm."

    try:
        # 1. Thực hiện tìm kiếm học thuật bằng Serper
        search_results = serper_search(query, num_results=10)
        
        # Kiểm tra lỗi
        if "error" in search_results:
            return f"Lỗi khi tìm kiếm: {search_results['error']}"
        
        # 2. Chuẩn bị ngữ cảnh cho Gemini
        scholar_data = ""
        sources = []
        
        if 'organic' in search_results:
            for i, result in enumerate(search_results['organic']):
                snippet = result.get('snippet', 'Không có mô tả')
                link = result.get('link', '#')
                title = result.get('title', 'Không có tiêu đề')
                
                scholar_data += f"[[{i+1}]] Tiêu đề: {title}\nĐoạn trích: {snippet}\nLink: {link}\n\n"
                sources.append({"index": i+1, "title": title, "link": link})

        if not scholar_data:
            return f"Agent không tìm thấy kết quả học thuật nào cho truy vấn: **{query}**."

        # 3. Tổng hợp kết quả bằng Gemini 
        system_prompt = (
            "Bạn là một chuyên gia phân tích dữ liệu nghiên cứu. Nhiệm vụ của bạn là tổng hợp "
            "thông tin từ các nguồn cung cấp, trả lời câu hỏi của người dùng một cách chi tiết, "
            "súc tích và hoàn toàn dựa trên các đoạn trích (snippets) bạn nhận được. "
            "Kết quả trả lời phải bằng tiếng Việt. "
            "Sau khi trả lời xong, bạn **PHẢI** cung cấp danh sách đầy đủ các nguồn đã trích dẫn theo định dạng Markdown."
        )

        user_prompt = (
            f"Sử dụng các đoạn trích dưới đây để trả lời câu hỏi: '{query}'\n\n"
            f"--- DỮ LIỆU TÌM KIẾM HỌC THUẬT ---\n\n{scholar_data}"
        )

        # VÒNG LẶP RETRY CHO GEMINI API
        response_text = None
        for attempt in range(max_retries):
            try:
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1
                    )
                )
                response_text = response.text
                break  
                
            except APIError as e:
                # Kiểm tra lỗi 503 hoặc 429
                if hasattr(e, 'response') and e.response.status_code in [503, 429]:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + 1  # 2s, 5s, 9s
                        print(f" Gemini quá tải (503/429). Thử lại sau {wait_time}s... (Lần {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        return f" Gemini API vẫn quá tải sau {max_retries} lần thử. Vui lòng thử lại sau vài phút."
                else:
                    # Lỗi khác không thể retry
                    raise e
        
        if response_text is None:
            return " Không thể tổng hợp kết quả sau nhiều lần thử."
        
        # 4. Định dạng đầu ra cuối cùng
        source_markdown = "\n\n** Có thể bạn muốn tham khảo thêm :**\n"
        for source in sources:
             source_markdown += f"- [[{source['index']}]] [{source['title']}]({source['link']})\n"
        
        return response_text + source_markdown
    
    except APIError as e:
        return f"Lỗi API Gemini trong Agent: {e}. Vui lòng kiểm tra GOOGLE_API_KEY hoặc thử lại sau."
    except Exception as e:
        return f"Lỗi không xác định khi chạy Agent: {e}"