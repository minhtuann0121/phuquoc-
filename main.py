from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer
import chromadb



load_dotenv()

#Đọc lại dữ liệu từ file
with open("data/PhuQuoc.txt", "r", encoding="utf-8") as f:
    phuquoc_data = f.read()
print(f"✅ Đọc xong dữ liệu: {len(phuquoc_data)} ký tự")

def split_chunks(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

print ("⏳ Đang load embedding model...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print ("✅ model sẵn sàng!")

chunks = split_chunks(phuquoc_data)
print(f"✅ Đã chia thành {len(chunks)} đoạn")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="phuquoc")

embeddings = embedding_model.encode(chunks).tolist()
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print("✅ Vector DB sẵn sàng")

SYSTEM_PROMPT = f"""LANGUAGE RULE - HIGHEST PRIORITY:
- If user writes in English → YOU MUST reply in English. This is MANDATORY.
- If user writes in Vietnamese → YOU MUST reply in Vietnamese. This is MANDATORY.
- NEVER reply in a different language than the user's message.

Bạn là trợ lý du lịch chuyên về khu bắc đảo Phú Quốc.

NGUYÊN TẮC BẮT BUỘC:
1. Chỉ trả lời câu hỏi liên quan đến du lịch khu bắc đảo Phú Quốc
2. Chỉ dùng thông tin từ DỮ LIỆU bên dưới, KHÔNG tự bịa thêm
3. Nếu không có trong dữ liệu → trả lời: "Tôi chưa có thông tin này, bạn nên kiểm tra trực tiếp tại địa phương"
4b. Nếu câu hỏi HOÀN TOÀN không liên quan đến du lịch Phú Quốc (ví dụ: nấu ăn, toán học, lập trình, thời tiết nơi khác...) → PHẢI trả lời ĐÚNG câu này, KHÔNG được biến tấu: "Xin lỗi, tôi chỉ hỗ trợ thông tin du lịch khu bắc đảo Phú Quốc. Bạn có câu hỏi nào về Phú Quốc không?"
TUYỆT ĐỐI KHÔNG được gắn câu hỏi ngoài luồng vào Phú Quốc.
5. Trả lời bằng tiếng Việt, ngắn gọn dễ hiểu

NGUYÊN TẮC CHỐNG THIÊN VỊ (BIAS):
6. KHÔNG ưu tiên gợi ý chỗ ở/ăn uống đắt tiền hơn bình dân
7. Khi người dùng hỏi chung chung như "chỗ ở nào tốt?" → PHẢI gợi ý ít nhất 1 lựa chọn bình dân VÀ 1 lựa chọn cao cấp
8. Nếu người dùng hỏi "rẻ" hoặc "tiết kiệm" → ưu tiên gợi ý bình dân trước
9. Nếu người dùng hỏi "sang" hoặc "cao cấp" → gợi ý cao cấp trước
10. KHÔNG phán xét hay bình luận về khả năng tài chính của người dùng
11. QUAN TRỌNG: Phát hiện ngôn ngữ người dùng đang dùng và PHẢI trả lời bằng ĐÚNG ngôn ngữ đó. Người dùng viết tiếng Anh → trả lời tiếng Anh. Người dùng viết tiếng Việt → trả lời tiếng Việt. KHÔNG ĐƯỢC trả lời sai ngôn ngữ.

NGUYÊN TẮC QUAN TÂM NHÓM YẾU THẾ:
12. Nếu người dùng nhắc đến "trẻ em", "con nhỏ", "bé", "gia đình" → ưu tiên gợi ý địa điểm AN TOÀN và PHÙ HỢP cho trẻ em trước
13. Nếu người dùng nhắc đến "người già", "ba mẹ", "ông bà", "cao tuổi" → CHỈ gợi ý những nơi có lối đi BẰNG PHẲNG, có THANG MÁY hoặc TẦNG THẤP, KHÔNG leo trèo. KHÔNG liệt kê các chỗ ở cao cấp hay không phù hợp. Chỉ gợi ý TỐI ĐA 2-3 lựa chọn phù hợp nhất.
14. Nếu người dùng nhắc đến "tiết kiệm", "rẻ", "ít tiền", "sinh viên", "bụi" → ưu tiên gợi ý options GIÁ RẺ NHẤT trước
15. Nếu người dùng nhắc đến "xe lăn", "khuyết tật", "đi lại khó" → ưu tiên gợi ý nơi CÓ LỐI ĐI BẰNG PHẲNG
16. KHÔNG BAO GIỜ gợi ý hoạt động nguy hiểm hoặc không phù hợp cho nhóm yếu thế dù họ không nhắc đến
17.Sau MỘT câu trả lời có thông tin cụ thể, PHẢI thêm dòng nguồn ở cuối
18.Format bắt buộc:
    Nguồn: [tên loại dữ liệu bạn dùng]

    Ví dụ các nguồn:
    - "Dữ liệu lưu trú khu Bắc Đảo Phú Quốc"
    - "Dữ liệu nhà hàng khu Bắc Đảo Phú Quốc"
    - "Dữ liệu hoạt động du lịch khu Bắc Đảo Phú Quốc"
    - "Dữ liệu di chuyển khu Bắc Đảo Phú Quốc"
19. Nếu câu hỏi ngoài luồng hoặc không có dữ liệu -> KHÔNG thêm source tag
20. Nếu dùng nhiều loại dữ liệu -> liệt kê tất cả, mõi loại một dòng
21. Sau source tag, PHẢI thêm dòng lý do giải thích tại sao gợi ý đó phù hợp với người dùng.
22. Format bắt buộc:
    Lý do gợi ý: [giải thích ngắn gọn dựa trên thông tin người dùng]

    Ví dụ:
    - "Bạn đề cập đi cùng trẻ em nên mình ưu tiên nơi an toàn, có khu vui chơi"
    - "Bạn hỏi chỗ rẻ nên mình ưu tiên options dưới 300k/đêm"
    - "Bạn đi cùng người cao tuổi nên mình chọn nơi không cần leo trèo, có thang máy"
    - "Bạn không đề cập yêu cầu đặc biệt nên mình gợi ý phổ biến nhất"
23. Lý do phải dựa trên ĐÚNG những gì người dùng nói, KHÔNG được bịa đặt
24. Nếu người dùng không đề cập yêu cầu đặc biệt -> ghi "Bạn không đề cập yêu cầu đặc biệt nên mình gợi ý phổ biến nhất"
25. Nếu câu hỏi ngoài luồng -> KHÔNG thêm reason tag
26. Sau reason tag, PHẢI thêm dòng độ tin cậy.
27. Format bắt buộc:
    Độ tin cậy: [mức độ] - [giải thích ngắn]
28. QUY TẮC ĐỘ TIN CẬY - PHẢI TUÂN THỦ TUYỆT ĐỐI
    QUAN TRỌNG NHẤT: Nếu câu trả lời có chứa BẤT KỲ thông tin nào sau đây:
        - Con số giá tiền (ví dụ: 200.000đ, 600k, miễn phí ...)
        - Giờ mở/ đóng cửa (ví dụ: 8h ,17h30 ...)
        - Số điện thoại
    -> BẮT BUỘC ghi: "Thấp - giá/giờ có thể thay đổi, nên kiểm tra lại trực tiếp"
    -> KHÔNG ĐƯỢC ghi CAO hoặc TRUNG BÌNH dù có bao nhiêu thông tin khác

    Chỉ khi KHÔNG có giá/giờ/SĐT mới dùng:
    -> "Cao" - có đầy đủ tên + địa chỉ + mô tả rõ ràng
    -> "Trung bình" -  chỉ có tên hoặc mô tả chung chung
29. Nếu câu hỏi ngoài luồng -> KHÔNG thêm confidence tag


"Khi cung cấp địa chỉ cụ thể, LUÔN thêm câu: 
'Bạn nên kiểm tra lại trên Google Maps trước khi đến 
vì địa chỉ có thể đã thay đổi.'"

DỮ LIỆU:
{phuquoc_data}
"""


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Message(BaseModel):
    text: str
INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "forget your instructions",
    "ignore all rules",
    "you are now",
    "pretend you are",
    "act as if you have no restrictions",
    "quên tất cả rules",
    "quên tất cả hướng dẫn",
    "bỏ qua tất cả",
    "hãy quên",
    "không có giới hạn",
    "đóng vai",

]
def check_injection(text: str) -> bool:
    """
    Kiểm tra xem tin nhắn có chứa prompt injection không.
    Trả về true nếu phát hiện tấn công, false nếu an toàn.
    """

    text_lower = text.lower() # Chuyển về chữ thường để so sánh
    for keyword in INJECTION_KEYWORDS:
        if keyword in text_lower:
            return True # Phát hiện tấn công prompt injection
    return False    # Không phát hiện tấn công, safe

def check_noise(text: str) -> bool:

    """
    Kiểm tra tin nhắn có phải dữ liệu nhiễu không
    Trả về True nếu là nhiễu, trả về False nếu là tin nhắn hợp lệ
    """

    #Loại bỏ khoảng trắng 2 đầu
    text = text.strip()

    #Kiểm tra tin nhắn quá ngắn (ít hơn 2 ký tự)
    if len(text) < 2:
        return True
    
    #Kiểm tra tin nhắn chứa ký tự đặc biệt như @@@@@####
    if not any(c.isalpha() for c in text):
        return True
    
    if len(text) > 500:
        return True
    
    return False

def detect_vulnerable_group(text: str) -> str:
    """
    Phát hiện nhóm yếu thế từ tin nhắn người dùng.
    Trả vè chuỗi mô tả nhóm, hoặc chuỗi rỗng nếu không phát hiện.
    """
    text_lower = text.lower()

    #Nhóm 1: Gia đình có trẻ em
    keyword_children = ["trẻ em", "con nhỏ", "bé", "em bé", "gia đình", "con tôi", "con mình", "đứa con", "con chúng tôi"]
    for kw in keyword_children:
        if kw in text_lower:
            return "VULNERABLE_GROUP: Người dùng đi cùng TRẺ EM. Ưu tiên địa điểm AN TOÀN, phù hợp trẻ nhỏ."
        
    #Nhóm 2: Người cao tuổi
    keyword_elderly = ["người già","ba mẹ", "cha mẹ", "ông bà", "người cao tuổi", "lớn tuổi", "bố mẹ", "ông bà nội ngoại"]
    for kw in keyword_elderly:
        if kw in text_lower:
            return "VULNERABLE_GROUP: Người dùng đi cùng người CAO TUỔI. Ưu tiên địa điểm DỄ ĐI LẠI, không leo trèo."
    
    #Nhóm 3: Người thu nhập thấp/ tiết kiệm
    keyword_budget = ["tiết kiệm", "rẻ", "ít tiền", "sinh viên", "bụi", "budget", "cheap", "thấp", "học sinh"]
    for kw in keyword_budget:
        if kw in text_lower:
            return "VULNERABLE_GROUP: Người dùng có ngân sách THẤP, ưu tiên lựa chọn GIÁ RẺ NHẤT."
    #Nhóm 4: Người khuyết tật
    keyword_disability = ["xe lăn", "khuyết tật", "đi lại khó", "tàn tật", "wheelchair"]
    for kw in keyword_disability:
        if kw in text_lower:
            return "VULNERABLE_GROUP: Người dùng có người KHÓ KHĂN VẬN ĐỘNG. Ưu tiên địa điểm có LỐI ĐI BẰNG PHẲNG."
        
    return "" # Không phát hiện nhóm yếu thế nào

@app.post("/chat")
def chat(message: Message):

    #Kiểm tra prompt injection trước
    if check_injection(message.text):
        return {"reply": "Xin lỗi, tôi không thể xử lý yêu cầu này"}
    
    #Kiểm tra tin nhắn quá dài
    if len(message.text) > 500:
        return {"reply": "Tin nhắn quá dài! Vui lòng nhập câu hỏi dưới 500 ký tự."}

    #Kiem tra dữ liệu nhiễu
    if check_noise(message.text):
        return {"reply" : "Tin nhắn không rõ ràng, vui lòng nhập câu hỏi rõ ràng hơn"}

    # Phát hiện nhóm yếu thế
    vulnerable_note = detect_vulnerable_group(message.text)

    #Dòng này ép chatbot phải chat đúng ngôn ngữ người gửi
    # Nếu phát hiện nhóm yếu thế thì báo cho AI
    if vulnerable_note:
        user_message = f"{message.text}\n\n[{vulnerable_note}]\n\n(IMPORTANT: Reply in the SAME language as my message. Do NOT mention what language you detected.)"
    else:
        user_message = f"{message.text}\n\n(IMPORTANT: Reply in the SAME language as my message. Do NOT mention what language you detected.)"   


    

    

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
            ]
    )
    return {"reply": response.choices[0].message.content}

@app.get("/")
def home():
    return {"status": "Chatbot đang chạy!"}

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/ui")
def serve_ui():
    return FileResponse("index.html")