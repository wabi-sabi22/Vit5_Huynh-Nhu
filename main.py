# main.py(fastapi) 
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import pdfplumber
from docx import Document
import io
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import os

# TÍCH HỢP PEFT 
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

#  TÍCH HỢP AGENT TỪ FILE KHÁC 
try:
    from search_agent import rag_agent, translate_text, gemini_client
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

#  Khởi tạo FastAPI App 
app = FastAPI(title="Chatbot Backend API")

#  Model Classes
class TextIn(BaseModel):
    text: str
    domain: str = "Y tế"

class QueryIn(BaseModel):
    query: str

class ApiResponse(BaseModel):
    result: str
    error: str = None

class BoolResponse(BaseModel):
    is_vietnamese: bool

#  Tải Model (Chạy 1 lần khi server khởi động) 
def load_model():
    model_path = r"E:\chatbotproject4\lora-lbc-fast_10k"
    BASE_MODEL_NAME = "VietAI/vit5-base" 

    if not PEFT_AVAILABLE:
        print("Bỏ qua load model cục bộ vì thiếu thư viện peft.")
    
    if PEFT_AVAILABLE and os.path.exists(model_path):
        try:
            print(f"Đang tải mô hình nền tảng ({BASE_MODEL_NAME}) và LoRA adapter...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            print("Đã load model gốc và LoRA adapter thành công!")
            return model, tokenizer
        except Exception as e:
            print(f"Lỗi khi load LoRA adapter: {e}. Đang chuyển sang load model online...")
    
    try:
        print("Đang tải model dự phòng (pengold) từ Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained("pengold/t5-vietnamese-summarization")
        model = AutoModelForSeq2SeqLM.from_pretrained("pengold/t5-vietnamese-summarization")
        print("Đã load model từ Hugging Face!")
        return model, tokenizer
    except Exception as e_online:
        print(f"Lỗi nghiêm trọng: Không thể load model tóm tắt dự phòng. Chi tiết: {e_online}")
        return None, None

#  Khởi tạo Model toàn cục 
model, tokenizer = load_model()
if model is None:
    print("CẢNH BÁO NGHIÊM TRỌNG: KHÔNG THỂ TẢI BẤT KỲ MODEL TÓM TẮT NÀO.")

#  Các hàm Logic (app.py)
def is_vietnamese(text: str) -> bool:
    try:
        if len(text) < 5: return False
        return detect(text.lower()) == "vi"
    except:
        return False

def summarize_text(text, domain="Y tế"):
    if not model or not tokenizer:
        return "Lỗi: Không tải được mô hình tóm tắt."
        
    domain_map = {
        "Công nghệ": "summarize_cong_nghe", "Khoa học": "summarize_khoa_hoc",
        "Y tế": "summarize_y_te", "Kinh tế": "summarize_kinh_te",
        "Xu hướng": "summarize_xu_huong", "Xã hội": "summarize_xa_hoi"
    }
    prefix = domain_map.get(domain, "summarize")

    input_text = f"{prefix}: {text}"
    inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to("cpu")
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=5)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_text_from_file_bytes(file_bytes: bytes, filename: str) -> str:
    filename = filename.lower()
    try:
        if filename.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = pdf.pages
                text_parts = [p.extract_text() for p in pages if p.extract_text()]
                return "\n".join(text_parts)
        elif filename.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            paras = [p.text for p in doc.paragraphs if p.text and p.text.strip() != ""]
            return "\n".join(paras)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return ""
    return ""

# Định nghĩa API Endpoints 
@app.post("/agent_search", response_model=ApiResponse)
async def agent_search_endpoint(data: QueryIn):
    if not AGENT_AVAILABLE:
        return JSONResponse(status_code=500, content={"result": "", "error": "Agent không khả dụng trên server."})
    try:
        result = rag_agent(data.query)
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": "", "error": str(e)})

@app.post("/extract_text", response_model=ApiResponse)
async def extract_text_endpoint(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        text = extract_text_from_file_bytes(file_bytes, file.filename)
        if not text:
            return JSONResponse(status_code=400, content={"result": "", "error": "Không thể trích xuất nội dung từ file."})
        return {"result": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": "", "error": str(e)})

@app.post("/detect_language", response_model=BoolResponse)
async def detect_language_endpoint(data: TextIn):
    result = is_vietnamese(data.text)
    return {"is_vietnamese": result}

@app.post("/translate", response_model=ApiResponse)
async def translate_endpoint(data: TextIn):
    if not AGENT_AVAILABLE:
        return JSONResponse(status_code=500, content={"result": "", "error": "Chức năng dịch không khả dụng."})
    try:
        result = translate_text(data.text, target_language='Vietnamese')
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": "", "error": str(e)})

@app.post("/summarize", response_model=ApiResponse)
async def summarize_endpoint(data: TextIn):
    if not model:
        return JSONResponse(status_code=500, content={"result": "", "error": "Model tóm tắt không khả dụng."})
    try:
        summary = summarize_text(data.text, data.domain)
        return {"result": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"result": "", "error": str(e)})

if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)