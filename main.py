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
import requests
import cohere
import re
from typing import List,Optional
import asyncio
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from scraper import scrape_and_save, is_data_outdated
from chunker import process_raw_file, clean_text, chunk_text
from reranker import rerank_places as ensemble_rerank, tinh_local_factor
from data.dia_diem import DIA_DIEM_BAC_DAO
from reranker import rerank_places
import json
import math



#Tạo thread pool dùng chung cho cả app
# max_workers=4 nghĩa là tối đa 4 workers chạy cùng lúc
executor = ThreadPoolExecutor(max_workers=4)
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
api_key = os.getenv("OPENWEATHER_API_KEY")
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
# Bước 1: INGESTION — đọc file vào bộ nhớ
#Đọc lại dữ liệu từ file

DULICH_FILE = "data/DuLich.txt"

if is_data_outdated(DULICH_FILE, days=7):
    # File cũ hơn 7 ngày hoặc chưa có -> scrape lại
    print("🔄 Dữ liệu cũ, đang cập nhật từ web...")
    dulich_data = scrape_and_save(DULICH_FILE)
else:
    #File còn mới -> đọc từ file luôn cho nhanh
    print("✅ Dữ liệu còn mới, đọc từ file...")
    with open(DULICH_FILE, "r", encoding="utf-8") as f:
        dulich_data = f.read()
print(f"✅ Đọc xong dữ liệu du lịch: {len(dulich_data)} ký tự")

#Phần đặt phòng giữ nguyên như cũ
with open("data/DatPhong.txt", "r", encoding="utf-8") as f:
    datphong_data = f.read()
print(f"✅ Đọc xong dữ liệu đặt phòng: {len(datphong_data)} ký tự")

# DuLich: dùng process_raw_file (tự clean + chunk theo marker === Nguồn ===)
dulich_chunks_raw = process_raw_file(DULICH_FILE, chunk_size=300, overlap=50)
dulich_chunks = [c["text"] for c in dulich_chunks_raw]
print(f"✅ Đã chia dữ liệu du lịch thành {len(dulich_chunks)} đoạn")

# DatPhong: không có marker === Nguồn ===, dùng chunk_text trực tiếp
datphong_cleaned = clean_text(datphong_data)
datphong_chunks_list = chunk_text(datphong_cleaned, chunk_size=300, overlap=50)
datphong_chunks = [c["text"] for c in datphong_chunks_list]
print(f"✅ Đã chia dữ liệu đặt phòng thành {len(datphong_chunks)} đoạn")


#Bước 3: EMBEDDING — chuyển chữ thành số để AI hiểu được
print ("⏳ Đang load embedding model...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print ("✅ model sẵn sàng!")


#Tạo vector DB với ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
existing_collections = [c.name for c in chroma_client.list_collections()]

if "dulich" not in existing_collections:
    print("⏳ Đang tạo Vector DB du lịch ...")
    dulich_collection = chroma_client.create_collection(name="dulich")
    dulich_embeddings = embedding_model.encode(dulich_chunks, show_progress_bar=True).tolist()
    dulich_collection.add(
        documents=dulich_chunks,
        embeddings=dulich_embeddings,
        ids=[f"dulich_{i}" for i in range(len(dulich_chunks))]
    )
    print("✅ Đã tạo Vector DB du lịch")
else:
    dulich_collection = chroma_client.get_collection(name="dulich")
    print("✅ Đã load Vector DB du lịch từ ổ đĩa!")

if "datphong" not in existing_collections:
    print("⏳ Đang tạo Vector DB đặt phòng ...")
    datphong_collection = chroma_client.create_collection(name="datphong")
    datphong_embeddings = embedding_model.encode(datphong_chunks, show_progress_bar=True).tolist()
    datphong_collection.add(
        documents=datphong_chunks,
        embeddings=datphong_embeddings,
        ids=[f"datphong_{i}" for i in range(len(datphong_chunks))]
    )
    print("✅ Đã tạo Vector DB đặt phòng")
else:
    datphong_collection = chroma_client.get_collection(name="datphong")
    print("✅ Đã load Vector DB đặt phòng từ ổ đĩa!")

def rerank(query: str, chunks: list, top_k: int = 3)->list:
    """
    Gửi (query + danh sách chunks) lên Cohere API.
    Cohere chấm điểm liên quan từng chunk, trả về top_k tốt nhất.  
    
    Khác với Cross-Encoder local:
    - Cross-Encoder: chạy trên máy bạn, miễn phí, chậm hơn
    - Cohere API: chạy trên server Cohere, tính phí (có free tier), nhanh và chính xác hơn
    """
    if not chunks:
        return chunks
    
    try:
        # Gửi request lên Cohere Rerank API
        response = co.rerank(
            model="rerank-v3.5", #Model rerank mới nhất của Cohere
            query=query,            # Câu hỏi người dùng
            documents=chunks,       # Danh sách chunks cần chấm điểm
            top_n=top_k,            # Chỉ lấy top_k kết quả tốt nhất
        )

        #Log điểm số để debug trong terminal
        for r in response.results:
            print(f"📄 Chunk #{r.index} | Score: {round(r.relevance_score,4)}")
        
        # Lấy lại nội dung chunk theo thứ tự Cohere đã sắp xếp
        # r.index là vị trí của chunk trong list 'chunk' ban đầu
        reranked_chunks = [chunks[r.index] for r in response.results]

        print(f"✅ Cohere rerank: giữ top {top_k} / {len(chunks)} chunks")
        return reranked_chunks
    
    except Exception as e:
        # Nếu API lõi (mất mạng, hết quota...) -> fallback về chunk gốc, không crash app
        print(f"⚠️ Cohere rerank thất bại: {e} -> dùng chunks gốc")
        return chunks[:top_k]

# Hàm tìm kiếm - Bước 5 RETRIEVAL 
def retrieve_dulich(query: str, top_k: int = 3) -> str:
    query_embedding = embedding_model.encode([query]).tolist()
    results = dulich_collection.query(query_embeddings=query_embedding, n_results=top_k * 2)
    candidates = results["documents"][0]

    reranked = rerank(query, candidates, top_k=top_k)
    return "\n---\n".join(reranked)

def retrieve_datphong(query: str, top_k: int = 3) -> str:
    query_embedding = embedding_model.encode([query]).tolist()
    results = datphong_collection.query(query_embeddings=query_embedding, n_results=top_k * 2)
    candidates = results["documents"][0]

    reranked = rerank(query, candidates, top_k=top_k)
    return "\n---\n".join(reranked)

with open("data/raw_osm.json", "r", encoding="utf-8") as f:
    OSM_PLACES = json.load(f)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # mét

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def search_places(user_lat, user_lon, top_k=5):
    places = []

    for place in OSM_PLACES:

        # Tính khoảng cách
        distance = haversine(
            user_lat,
            user_lon,
            place["lat"],
            place["lon"]
        )

        # Fake data tạm thời để test ensemble
        place["distance_meters"] = distance
        place["rating"] = 4.2
        place["price_level"] = 1
        place["is_open_now"] = 1

        # local_factor demo
        place["ho_kinh_doanh_dia_phuong"] = True
        place["chu_la_ngu_dan"] = False
        place["nghe_truyen_thong"] = False
        place["dung_lao_dong_yeu_the"] = False

        places.append(place)

    ranked = rerank_places(places)
    
    for p in ranked[:5]:
        print(p["ten"], p["final_score"])
    return ranked[:top_k]


SYSTEM_PROMPT_DULICH = """LANGUAGE RULE - HIGHEST PRIORITY:
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
    📌 Nguồn: [tên loại dữ liệu bạn dùng]

    Ví dụ các nguồn:
    - "📌 Nguồn: Dữ liệu lưu trú khu Bắc Đảo Phú Quốc"
    - "📌 Nguồn: Dữ liệu nhà hàng khu Bắc Đảo Phú Quốc"
    - "📌 Nguồn: Dữ liệu hoạt động du lịch khu Bắc Đảo Phú Quốc"
    - "📌 Nguồn: Dữ liệu di chuyển khu Bắc Đảo Phú Quốc"
19. Nếu câu hỏi ngoài luồng hoặc không có dữ liệu -> KHÔNG thêm source tag
20. Nếu dùng nhiều loại dữ liệu -> liệt kê tất cả, mõi loại một dòng
21. Sau source tag, PHẢI thêm dòng lý do giải thích tại sao gợi ý đó phù hợp với người dùng.
22. Format bắt buộc:

    - Mỗi địa điểm / lựa chọn PHẢI xuống dòng riêng, dùng gạch đầu dòng (-)
    - KHÔNG được viết nhiều địa điểm trên cùng một dòng
    - Ví dụ đúng:
        - Địa điểm A: mô tả ngắn
        - Địa điểm B: mô tả ngắn
    💡 Lý do gợi ý: [giải thích ngắn gọn dựa trên thông tin người dùng]

    Ví dụ:
    - "💡 Lý do gợi ý: Bạn đề cập đi cùng trẻ em nên mình ưu tiên nơi an toàn, có khu vui chơi"
    - "💡 Lý do gợi ý: Bạn hỏi chỗ rẻ nên mình ưu tiên options dưới 300k/đêm"
    - "💡 Lý do gợi ý: Bạn đi cùng người cao tuổi nên mình chọn nơi không cần leo trèo, có thang máy"
    - "💡 Lý do gợi ý: Bạn không đề cập yêu cầu đặc biệt nên mình gợi ý phổ biến nhất"
23. Lý do phải dựa trên ĐÚNG những gì người dùng nói, KHÔNG được bịa đặt
24. Nếu người dùng không đề cập yêu cầu đặc biệt -> ghi "Bạn không đề cập yêu cầu đặc biệt nên mình gợi ý phổ biến nhất"
25. Nếu câu hỏi ngoài luồng -> KHÔNG thêm reason tag
26. Sau reason tag, PHẢI thêm dòng độ tin cậy.
27. Format bắt buộc:
    🎯 Độ tin cậy: [mức độ] - [giải thích ngắn]
28. QUY TẮC ĐỘ TIN CẬY - PHẢI TUÂN THỦ TUYỆT ĐỐI
    QUAN TRỌNG NHẤT: Nếu câu trả lời có chứa BẤT KỲ thông tin nào sau đây:
        - Con số giá tiền (ví dụ: 200.000đ, 600k, miễn phí ...)
        - Giờ mở/ đóng cửa (ví dụ: 8h ,17h30 ...)
        - Số điện thoại
    -> 🎯 Độ tin cậy: BẮT BUỘC ghi: "Thấp - giá/giờ có thể thay đổi, nên kiểm tra lại trực tiếp"
    -> KHÔNG ĐƯỢC ghi CAO hoặc TRUNG BÌNH dù có bao nhiêu thông tin khác

    Chỉ khi KHÔNG có giá/giờ/SĐT mới dùng:
    -> 🎯 Độ tin cậy:  "Cao" - có đầy đủ tên + địa chỉ + mô tả rõ ràng
    -> 🎯 Độ tin cậy:  "Trung bình" -  chỉ có tên hoặc mô tả chung chung
29. Nếu câu hỏi ngoài luồng -> KHÔNG thêm confidence tag


"Khi cung cấp địa chỉ cụ thể, LUÔN thêm câu: 
🗺️'Bạn nên kiểm tra lại trên Google Maps trước khi đến vì địa chỉ có thể đã thay đổi.'"


"""

SYSTEM_PROMPT_DATPHONG = """LANGUAGE RULE - HIGHEST PRIORITY:
- If user writes in English -> YOU MUST reply in English. This is MANDATORY.
- If user writes in Vietnamese -> YOU MUST reply in Vietnamese. This is MANDATORY.

Bạn là trợ lý tư vấn chỗ ở chuyên về khu bắc đảo Phú Quốc.

NGUYÊN TẮC FORMAT:
- Mỗi địa điểm / lựa chọn PHẢI xuống dòng riêng, dùng gạch đầu dòng (-)
- KHÔNG được viết nhiều địa điểm trên cùng một dòng
- Ví dụ đúng:
    - Địa điểm A: mô tả ngắn
    - Địa điểm B: mô tả ngắn

NGUYÊN TẮC BẮT BUỘC:
1. Chỉ trả lời câu hỏi liên quan đến chỗ ở, phòng, khách sạn, homestay tại Phú Quốc
2. Chỉ dùng thông tin từ DỮ LIỆU bên dưới, KHÔNG bịa đặt thêm
3. Nếu không có trong dữ liệu -> trả lời: "Tôi chưa có thông tin này, bạn nên kiểm tra trực tiếp tại địa phương"
4. KHÔNG ưu tiên gợi ý chỗ đắt tiền hơn bình dân
5. Khi hỏi chung chung -> gợi ý ít nhất 1 bình dân và 1 cao cấp
6. "rẻ/ tiết kiệm/ sinh viên" -> ưu tiên bình dân trước
7. "sang/ cao cấp" -> ưu tiên cao cấp trước
8. "người già/ cao tuổi" -> chỉ gợi ý nơi có thang máy, tầng thấp, dễ đi lại
9. "trẻ em/ gia đình" -> ưu tiên nơi có hồ bơi trẻ em, an toàn

NGUYÊN TẮC MINH BẠCH:
10. Thêm dòng nguồn:📌 Nguồn: Dữ liệu chỗ ở khu Bắc Đảo Phú Quốc
11. Thêm lý do gợi ý:💡 Lý do gợi ý: [giải thích ngắn]
12. Thêm độ tin cậy:
    - Có giá tiền ->🎯 Độ tin cậy: "Thấp - giá có thể thay đổi, nên kiểm tra lại"
    - Không có giá, đủ tên + địa chỉ ->🎯 Độ tin cậy: "Cao"
13. Khi cung cấp địa chỉ -> thêm: 🗺️"Bạn nên kiểm tra lại trên Google Maps trước khi đến"
"""

SYSTEM_PROMPT_THOITIET = """ LANGUAGE RULE - HIGHEST PRIORITY: 
- If user writes in English -> YOU MUST reply in English. This is MANDATORY.
- If user writes in Vietnamese -> YOU MUST reply in Vietnamese. This is MANDATORY.

Bạn là trợ lý tư vấn thời tiết ở Phú Quốc.

FORMAT BẮT BUỘC - PHẢI TUÂN THỦ CHÍNH XÁC:

### Thời tiết ở Phú Quốc hôm nay ([Ngày từ dữ liệu])
[1 câu mô tả chung thời tiết Phú Quốc hiện tại - nắng/mây/mưa]

• Ghi nhận lúc [Giờ ghi nhận]: nhiệt độ [X]°C, cảm giác thực tế [Y]°C do độ ẩm [Z]%
• [Đánh giá sóng biển từ dữ liệu - copy nguyên câu từ trường "Đánh giá sóng biển"]


NGUYÊN TẮC:
- Khi thời tiết xấu/sóng lớn → gợi ý tham quan trong nhà, chụp ảnh tại khách sạn
- Khi có bão → khuyên chờ hết bão mới du lịch
- Ngắn gọn, thực tế, KHÔNG giải thích dài dòng
- KHÔNG thêm tag 📌💡🎯 vào câu trả lời thời tiết

"""


def worker_dulich(query, vulnerable_note, history=[]):

        

 

    try:
        # Bước 1:  Tìm dữ liệu du lịch liên quan
        retrieved_context = retrieve_dulich(query, top_k=3)

        import copy
        dia_diem = copy.deepcopy(DIA_DIEM_BAC_DAO)
        
        # Tính local_factor cho từng địa điểm
        for d in dia_diem:
            d["local_factor"] = tinh_local_factor(d)
    
        
        # Rerank
        rerank_preference = detect_rerank_preference(query)
        top_dia_diem = ensemble_rerank(dia_diem, preference=rerank_preference)
        
        # Tạo text gợi ý để đưa vào prompt, kèm trace Bagging/Boosting để AI giải thích được.
        goi_y_lines = []
        for d in top_dia_diem:
            breakdown = d.get("score_breakdown", {})
            goi_y_lines.append(
                f"- Hạng {d.get('rank')}: {d['ten']}"
                f" | FinalScore: {d['final_score']}"
                f" | Bagging: locality={breakdown.get('tree1_locality')}, proximity={breakdown.get('tree2_proximity')}, quality={breakdown.get('tree3_quality')}, s_bag={breakdown.get('s_bag')}"
                f" | Boosting: fairness={breakdown.get('delta1_fairness')}, accessibility={breakdown.get('delta2_access')}"
                f" | UserPreference: {breakdown.get('preference')} bonus={breakdown.get('preference_bonus')}"
                f" | Địa phương: {'✅' if d.get('local_factor', 0) > 0.3 else '❌ chuỗi lớn'} (local_factor={d.get('local_factor')})"
                f" | Giá: {'rẻ' if d.get('price_level', 2) <= 1 else 'trung bình' if d.get('price_level', 2) <= 2 else 'đắt'}"
                f" | {d['mo_ta']}"
            )
        goi_y_text = "\n".join(goi_y_lines)

        # Bước 2: Ghép câu trả lời + dữ liệu + gửi cho AI
        user_message = (
            f"{query}\n\n"
            + (f"{vulnerable_note}\n\n" if vulnerable_note else "")
            + f"DỮ LIỆU LIÊN QUAN:\n{retrieved_context}\n\n"
            + f"DANH SÁCH ĐỊA ĐIỂM GỢI Ý (preference={rerank_preference}, đã xếp hạng):\n{goi_y_text}\n\n"
            + "QUAN TRỌNG: Nếu người dùng hỏi về địa điểm ăn uống, hãy dùng DANH SÁCH ĐỊA ĐIỂM GỢI Ý ở trên để trả lời.\n\n"
            + "QUAN TRỌNG: Khi so sánh địa điểm, hãy giải thích rõ:\n"
            + "- Quán nào được hệ thống ưu tiên hơn và tại sao (dựa vào FinalScore, preference bonus, địa phương, giá cả)\n"
            + "- Quán địa phương (✅) được ưu tiên hơn chuỗi lớn (❌) vì hỗ trợ tiểu thương bản địa\n\n"
            + "(IMPORTANT: Reply in SAME language as my message.)"
        )
        # Chuyển history thành list dict cho Groq
        history_messages = [{"role" : h.role, "content": h.content} for h in history]

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            timeout=30,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DULICH},
                *history_messages,                           # <- các lượt trước
                {"role": "user", "content": user_message}   # <- lượt hiện tại
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ Groq API lỗi (dulich): {e}")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
    
def worker_datphong(query, vulnerable_note, history=[]):
    try:
        retrieved_context = retrieve_datphong(query, top_k=3)
        user_message = (
            f"{query}\n\n"
            + (f"{vulnerable_note}\n\n" if vulnerable_note else "")
            + f"DỮ LIỆU LIÊN QUAN:\n{retrieved_context}\n\n"
            + "(IMPORTANT: Reply in SAME language as my message.)"
        )
        history_messages = [{"role" : h.role, "content" : h.content} for h in history]
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            timeout=30,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DATPHONG},
                *history_messages,
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ Groq API lỗi (datphong): {e}")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."

def worker_thoitiet(query, vulnerable_note, history=[]):
    try:
        weather_response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=10.2899&lon=103.9840&appid={api_key}&units=metric", timeout = 10)
        data = weather_response.json()
        nhiet_do = data["main"]["temp"]
        nhiet_do_cam_giac = data["main"]["feels_like"]
        do_am = data["main"]["humidity"]
        toc_do_gio = data["wind"]["speed"]
        mo_ta_thoi_tiet = data["weather"][0]["description"]

        
        vn_tz = timezone(timedelta(hours=7))
        gio_hien_tai = datetime.now(vn_tz).strftime("%H: %M")
        ngay_hien_tai = datetime.now(vn_tz).strftime("%d/%m/%Y")

        # Đánh giá sóng biển dựa vào tốc độ gió
        if toc_do_gio < 3:
            song_bien = "Biển rất êm, sóng nhỏ, lý tưởng để tắm biển và lặn"
        elif toc_do_gio < 6:
            song_bien = "Biển tưởng đối êm, sóng nhẹ, phù hợp tắm biển"
        elif toc_do_gio <10:
            song_bien = "Biển có sóng vừa, cần thận trọng khi tắm, không nên đi xa bờ"
        else:
            song_bien = "Biển động, sóng lớn, KHÔNG nên tắm biển hoặc đi thuyền"

        user_message = (
            f"{query}\n\n"
            + (f"{vulnerable_note}\n\n" if  vulnerable_note else "")
            + f"DỮ LIỆU LIÊN QUAN:\n"
            + f"Ngày: {ngay_hien_tai}\n"
            + f"Giờ ghi nhận: {gio_hien_tai}\n"
            + f"Nhiệt độ: {nhiet_do}°C\n"
            + f"Cảm giác thực tế: {nhiet_do_cam_giac}°C\n"
            + f"Độ ẩm: {do_am}%\n"
            + f"Tốc độ gió: {toc_do_gio}km/h\n"
            + f"Mô tả thời tiết: {mo_ta_thoi_tiet}\n"
            + f"Đánh giá sóng biển: {song_bien}\n\n"
            + "(IMPORTANT: Reply in SAME language as my message.)"
        )
        history_messages =[{"role" : h.role, "content" : h.content} for h in history]
        groq_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            timeout = 30,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_THOITIET},
                *history_messages,
                {"role": "user", "content": user_message}
            ]
        )
        return groq_response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ Groq API lỗi (thoitiet): {e}")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
def worker_unknown():
    return "Xin lỗi, tôi chỉ hỗ trợ thông tin du lịch khu bắc đảo Phú Quốc. Bạn có câu hỏi nào về Phú Quốc không?"

def merger_agent(query: str, results: dict) -> str:
    """
    Hàm này nhận kết quả từ nhiều worker và ghép thành 1 câu trả lời.

    Tham số:
    - query: câu hỏi gốc của người dùng
    - results: dict chứa kết quả từng worker
    Ví dụ:{
        "datphong": "Mình gợi ý khách sạn ABC...",
        "thoitiet": "Hôm nay trời đẹp, nhiệt độ 30 độ..."
    }

    Tại sao cần merger?
    Nếu không có merger, bạn chỉ có thể nối chuỗi thô:
        "Mình gợi ý khách sạn ABC... \n\nHôm nay trời đẹp..."
    -> Trông rất kỳ, không tự nhiên, có thể trùng lập thông tin.

    Merger dùng LLM để viết lại thành 1 đoạn liền mạch.
    """

    valid_results = {k: v for k, v in results.items() if k != "unknown"}
    has_unknown = "unknown" in results

    if not valid_results:
        return worker_unknown()

    # Nếu chỉ có 1 worker chạy -> không cần merger, trả thẳng
    # Tiết kiệm 1 lần gọi API không cần thiết
    if len(valid_results) == 1:
        print("⚡ Chỉ 1 agent → bỏ qua merger")
        result = list(valid_results.values())[0]
    else:
    # Chuẩn bị nội dung để gửi cho merger
    # Đặt nhãn rõ ràng để LLM hiểu phần nào từ agent nào
        label_map = {
            "dulich": "=== Thông tin du lịch ===",
            "datphong": "=== Thông tin đặt phòng ===",
            "thoitiet": "=== Thông tin thời tiết ===",
                    }

        parts = []
        for agent_name, content in valid_results.items():
            label = label_map.get(agent_name, f"=== {agent_name} ===")
            parts.append(f"{label}\n{content}")
        
        combined_text = "\n\n".join(parts)

        print(f"🔀 merger đang ghép {len(valid_results)} kết quả...")

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                timeout = 30,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý tổng hợp thông tin du lịch Phú Quốc.\n"
                            "Bạn nhận được câu trả lời từ nhiều chuyên gia khác nhau.\n"
                            "NHIỆM VỤ: ghép chúng thành 1 câu trả lời DUY NHẤT, tự nhiên.\n\n"
                            "NGUYÊN TẮC BẮT BUỘC:\n"
                            "1. Giữ TOÀN BỘ thông tin quan trọng, không bỏ sót\n"
                            "2. KHÔNG lặp lại thông tin giống nhau\n"
                            "3. Dùng tiêu đề phân chia rõ nếu có nhiều chủ đề khác nhau\n"
                            "4. GIỮ NGUYÊN các tag: 📌 Nguồn, 💡 Lý do, 🎯 Độ tin cậy, 🗺️\n"
                            "5. Trả lời bằng CÙNG ngôn ngữ với câu hỏi gốc\n"
                            "6. Mở đầu tự nhiên, KHÔNG bắt đầu bằng 'Dưới đây tổng hợp...'"
                            "7. Các tag 📌 Nguồn, 💡 Lý do, 🎯 Độ tin cậy PHẢI đặt ở CUỐI CÙNG, sau toàn bộ nội dung, KHÔNG được đặt giữa các đoạn\n"
                            "8. TUYỆT ĐỐI KHÔNG tự thêm thông tin ngoài những gì chuyên gia cung cấp. "
                            "   Nếu chuyên gia không có thông tin về một chủ đề -> KHÔNG được tự bịa. "
                            "   Chỉ tổng hợp đúng những gì có trong dữ liệu được cung cấp.\n"
                            "9. QUAN TRỌNG - FORMAT DANH SÁCH: Mỗi khách sạn/địa điểm/lựa chọn PHẢI xuống dòng riêng với gạch đầu dòng (-)\n"
                            "   KHÔNG được viết nhiều địa điểm trên cùng 1 đoạn văn liền\n"
                            "   Ví dụ đúng:\n"
                            "   - Khách sạn A: giá 200k/đêm\n"
                            "   - Khách sạn B: giá 300k/đêm\n"
                            "   Ví dụ SAI: 'Khách sạn A giá 200k và khách sạn B giá 300k...'\n"
                            "10. TUYỆT ĐỐI KHÔNG tự thêm thông tin ngoài những gì chuyên gia cung cấp\n"
                            "   KHÔNG trả lời các câu hỏi không có trong dữ liệu chuyên gia\n"
                            "   Nếu một phần câu hỏi không được chuyên gia trả lời -> BỎ QUA hoàn toàn, KHÔNG tự bịa\n"
                            "11. KHÔNG dùng markdown headers (###), thay bằng dòng in hoa hoặc emoji\n"
                            "12. KHÔNG dùng văn xuôi để mô tả danh sách — PHẢI dùng bullet points\n"
                            "13. QUAN TRỌNG - FORMAT DANH SÁCH: Mỗi thời gian ghi nhận, nhiệt độ, cám giác thực tế là bao nhiêu và do độ ẩm/ đánh giá sóng biển PHẢI xuống dòng riêng với chấm tròn đầu dòng (•)\n"
                            "   Không được viết thời gian ghi nhận, nhiệt độ, cám giác thực tế là bao nhiêu và do độ ẩm/ đánh giá sóng biển trên cùng 1 đoạn văn liền\n"
                            "   Ví dụ ĐÚNG:\n"
                            "   • Ghi nhận lúc 11:05: nhiệt độ 32.57°C, cảm giác thực tế 38.44°C do độ ẩm 60%\n"
                            "   • Biển có sóng vừa, cần thận trọng khi tắm, không nên đi xa bờ"
                            "   Ví dụ SAI: 'Ghi nhận lúc 11:05: nhiệt độ 32.57°C, cảm giác thực tế 38.44°C do độ ẩm 60%. Biển có sóng vừa, cần thận trọng khi tắm, không nên đi xa bờ...'\n"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Câu hỏi gốc của người dùng:\n{query}\n\n"
                            f"Câu trả lời từ các chuyên gia:\n\n{combined_text}"
                        )
                    }
                ]
            )
            result = response.choices[0].message.content
            print("✅ Merger hoàn thành")
        except Exception as e:
            # Nếu merger fail -> ghép thủ công, không crash app
            print(f"⚠️ Merger lỗi: {e} -> fallback ghép thủ công")
            result = combined_text

    if has_unknown:
        result += "\n\n*(Lưu ý: Một số câu hỏi nằm ngoài phạm vi hỗ trợ của tôi — tôi chỉ tư vấn du lịch khu bắc đảo Phú Quốc.)*"

    return result


#Supervisor dùng AI để phân loại
def supervisor_ai(text: str, history: list = []) -> list:
    """
    THAY ĐỔI SO VỚI CŨ:
    - Cũ: trả về 1 string -> "dulich"
    - Mới: trả về 1 string -> ["dulich" "thoitiet"]

    Tại sao dùng list?
    Vì câu hỏi liên quan nhiều chủ đề cùng lúc.
    Ví dụ: "Khách sạn nào gần biển, hôm nay có bão không?" 
    -> cần cả datphong và thoitiet
    """
    history_messages = [{"role": h.role, "content": h.content} for h in history]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,      # =0 để kết quả nhất quán
        timeout = 30,
        max_tokens=20,      # chỉ cần vài từ
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là bộ phận phân loại câu hỏi cho chatbot du lịch Phú Quốc. \n"
                    "Phân loại câu hỏi MỘT HOẶC NHIỀU loại:\n"
                    "- dulich: địa điểm tham quan, ăn uống, nhà hàng, hoạt động, di chuyển\n"
                    "- datphong: chỗ ở, khách sạn, homestay, resort, giá phòng, đặt phòng\n"
                    "- thoitiet: nhiệt độ, gió, mưa, thời tiết, có nên đi biển không\n"
                    "- unknown: không liên quan đến Phú Quốc hoặc du lịch\n\n"
                    "CÁCH TRẢ LỜI:\n"
                    "- Nếu 1 chủ đề: chỉ viết từ đó. Ví dụ: dulich\n"
                    "- Nếu nhiều chủ đề: viết cách nhau bằng dấu phẩy. Ví dụ: datphong, thoitiet\n"
                    "- Nếu unknown: chỉ viết unknown (không kết hợp với gì khác)\n"
                    "- Nếu câu hỏi có NHIỀU VẾ, một số thuộc Phú Quốc, một số không:\n"
                    "  → Chỉ liệt kê các vế THUỘC Phú Quốc, BỎ QUA hoàn toàn vế ngoài luồng\n"
                    "  → Ví dụ: 'khách sạn rẻ + cách nấu cơm' → chỉ trả về: datphong\n"
                    "KHÔNG giải thích, KHÔNG thêm gì khác."
                )
            },
            *history_messages,
            {"role": "user", "content": text}
        ]
    )

    raw = response.choices[0].message.content.strip().lower()
    print(f"🧠 Supervisor raw: '{raw}'")

    #Parse kết quả thành list, lọc bỏ những từ không hợp lệ
    valid_agents = {"dulich", "datphong", "thoitiet"}
    parsed = [a.strip() for a in raw.split(",") if a.strip() in valid_agents]

    # Nếu AI trả về "unknown" hoặc không parse được gì -> unknown
    if not parsed:
        return["unknown"]
    
    print(f"🧠 Supervisor chọn: {parsed}")
    return parsed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class HistoryItem (BaseModel):
    role: str  # "user" hoặc "assistant"
    content: str

class Message(BaseModel):
    text: str
    history: Optional[List[HistoryItem]] = [] # mặc định rỗng nếu không gửi
    
def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text trước khi kiểm tra injection:
    - Bỏ dấu cách thừa giữa các ký tự: "i g n o r e" ->"ignore"
    - Chuyển unicode về ASCII: "ｉｇｎｏｒｅ" -> "ignore"
    - Chuyển về cỡ chữ thường
    """
    #Chuẩn hóa unicode fullwidth ->ASCII
    text = unicodedata.normalize('NFKC', text)
    # Xóa khoảng trắng thừa giữa từng ký tự (i g n o r e -> ignore)
    # Chỉ áp dụng khi mỗi "từ" chỉ có 1 ký tự
    words = text.split()
    if all(len(w) == 1 for w in words[:6]):
        text = ''.join(words)
    return text.lower()

INJECTION_KEYWORDS = [
    # Tiếng Anh
    "ignore previous instructions",
    "forget your instructions",
    "ignore all rules",
    "you are now",
    "pretend you are",
    "act as if you have no restrictions",
    "disregard your instructions",
    "override your instructions",
    "new instructions",
    "system promt:",
    # Tiếng việt
    "quên tất cả rules",
    "quên tất cả hướng dẫn",
    "bỏ qua tất cả",
    "hãy quên",
    "không có giới hạn",
    "đóng vai",
    "giả vờ là",
    "bỏ qua hướng dẫn",
    "làm theo lệnh mới",
]

def check_injection(text: str) -> bool:
    """
    Kiểm tra xem tin nhắn có chứa prompt injection không.
    Trả về true nếu phát hiện tấn công, false nếu an toàn.
    3 lớp kiểm tra:
    1. Normalized text lowercase (chống bypass bằng khoảng trống/ unicode) 
    2. Original text lowercase (chống bypass thông thường)
    3. Text không dấu cách (chống "i.g.n.o.r.e")
    """
    normalized = normalize_text(text)
    original_lower = text.lower()
    no_spaces = original_lower.replace(' ', '').replace('.', '').replace('-', '')

    for keyword in INJECTION_KEYWORDS:
        keyword_no_space = keyword.replace(' ','')
        if (keyword in normalized
            or keyword in original_lower
            or keyword_no_space in no_spaces):
            print(f"🚨 Injection detected: '{keyword}'")
            return True
    return False

def check_noise(text: str) -> bool:

    """
    Kiểm tra tin nhắn có phải dữ liệu nhiễu không
    Trả về True nếu là nhiễu, trả về False nếu là tin nhắn hợp lệ
    """

    text = text.strip()

    #Quá ngắn
    if len(text) < 2:
        return True

    #Không có chữ cái nào
    if not any(c.isalpha() for c in text):
        return True

    # Lặp ký tự quá nhiều: "aaaaaaaaaaaaa", "????????????????"
    if len(set(text.replace(' ',''))) < 3 and len(text) > 6:
        return True
    
    #Toàn số và ký tự đặc biệt, không có nghĩa
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.3 and len(text) > 10:
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
def detect_rerank_preference(text: str) -> str:
    """Detect explicit ranking preference from the chat message/UI clarification."""
    text_lower = text.lower()

    if any(kw in text_lower for kw in ["giá rẻ", "rẻ", "tiết kiệm", "gia re", "tiet kiem", "cheap", "budget"]):
        return "cheap"
    if any(kw in text_lower for kw in ["địa phương", "bản địa", "truyền thống", "dia phuong", "ban dia", "truyen thong", "local"]):
        return "local"
    if any(kw in text_lower for kw in ["gần", "gần nhất", "khoảng cách", "gan", "khoang cach", "near", "nearest"]):
        return "near"
    return "balanced"

def run_worker_sync(agent: str, query: str, vulnerable_note: str, history: list) -> tuple:
    """
    Hàm wrapper để chạy worker trong thread pool.

    Tại sao cần wrapper này?
    asyncio.gather() cần các coroutine (async function).
    Nhưng worker_dulich, worker_datphong... là hàm thường (sync).
    -> Dùng run_in_executor để chạy hàm sync trong thread riêng,
        asyncio.gather() có thể chờ tất cả cùng lúc.
    Trả về tuple (tên_agent, kết_quả) để biết kết quả từ agent nào.
    """
    if agent == "dulich":
        result = worker_dulich(query, vulnerable_note, history)
        return("dulich", result)
    elif agent == "datphong":
        result = worker_datphong(query, vulnerable_note, history)
        return("datphong", result)
    elif agent == "thoitiet":
        result = worker_thoitiet(query, vulnerable_note, history)
        return("thoitiet", result)
    else:
        return ("unknown", worker_unknown())

@app.post("/chat")
async def chat(message: Message): #<- THÊM async vào đây
    """
    Thay đổi so với cũ:
    1. async def thay vì def -> FastAPI chạy bất đồng bộ
    2. Supervisor trả về list thay vì string
    3. Các worker chạy song song thay vì nối tiếp
    4. Merger ghép kết quả lại
    """

    # Các bước kiểm tra an toàn giữ nguyên 
    if check_injection(message.text):
        return {"reply": "Xin lỗi, tôi không thể xử lý yêu cầu này"}
    if len(message.text) > 500:
        return {"reply": "Tin nhắn quá dài! Vui lòng nhập câu hỏi dưới 500 ký tự."}
    if check_noise(message.text):
        return{"reply": "Tin nhắn không rõ ràng, vui lòng nhập câu hỏi rõ ràng hơn"}
    
    vulnerable_note = detect_vulnerable_group(message.text)
    history = message.history or []

    # Bước mới 1: Supervisor trả về LIST
    agents = supervisor_ai(message.text, history)
    # agents có thể là: ["dulich"], ["datphong", "thoitiet"], v.v.

    if agents == ["unknown"]:
        return {"reply": worker_unknown()}
    
    # Bước mới 2: chạy các worker SONG SONG
    # Không dùng asyncio.gather() trực tiếp với hàm sync,
    # mà dùng run_in_executor để đưa vào thread pool
    #
    # Hình dung như sau:
    # - Cũ: gọi worker A → chờ xong → gọi worker B → chờ xong → tổng hợp
    # - Mới: gọi worker A và B cùng lúc → chờ cả 2 xong → tổng hợp
    #
    # Nếu mỗi worker mất 2 giây:
    # - Cũ: 2 + 2 = 4 giây
    # - Mới: max(2, 2) = 2 giây  ← nhanh gấp đôi

    loop = asyncio.get_running_loop()

    tasks = [
        loop.run_in_executor(
            executor,           #thread pool đã tạo
            run_worker_sync,    #hàm cần chạy
            agent,              #tham số 1
            message.text,       #tham số 2
            vulnerable_note,    #tham số 3
            history             #tham số 4
        )
        for agent in agents     #tạo 1 task cho mỗi agent trong list
    ]
    
    # chờ TẤT CẢ task hoàn thành cùng lúc
    results_list = await asyncio.gather(*tasks)
    # results_list = [("dulich", "kết quả..."), ("thoitiet", "kết quả...")]

    # Chuyển list of tuples -> dict để dễ xử lý 
    results_dict = {agent_name: content for agent_name, content in results_list}
    print(f"✅ Nhận được kết quả từ: {list(results_dict.keys())}")

    # Bước mới 3: Merger tổng hợp
    final_reply = merger_agent(message.text, results_dict)

    return {"reply": final_reply}


@app.get("/")
def home():
    return {"status": "Chatbot đang chạy!"}

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/ui")
def serve_ui():
    return FileResponse("index.html")

