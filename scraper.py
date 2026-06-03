import requests # gửi requests lên internet
import time     # để dừng 1 giây giữa các request (theo robots.txt)
import os       # để kiểm tràile có tồn tại không
from datetime import datetime, timedelta # để kiểm tra file có cũ không
from cleaner import clean_text


# -------------------------------------------------------
# DANH SÁCH CÁC TRANG SẼ SCRAPE
# Mỗi trang là 1 URL của visitphuquoc.com.vn
# -------------------------------------------------------

URLS = [
    "https://visitphuquoc.com.vn/vi",                              # Trang chủ
    "https://visitphuquoc.com.vn/vi/xem-gi-lam-gi",               # Xem gì làm gì
    "https://visitphuquoc.com.vn/vi/an-uong-phu-quoc",            # Ăn uống
    "https://visitphuquoc.com.vn/vi/mua-sam-phu-quoc",            # Mua sắm
    "https://visitphuquoc.com.vn/vi/noi-o",                       # Nơi ở
    "https://visitphuquoc.com.vn/vi/thong-tin-du-lich-can-biet",  # Thông tin cần biết
    "https://visitphuquoc.com.vn/vi/diem-den-duoc-yeu-thich-tai-phu-quoc", # Điểm đến
    "https://visitphuquoc.com.vn/vi/di-chuyen-den-phu-quoc",      # Di chuyển
    "https://visitphuquoc.com.vn/vi/di-chuyen-quanh-phu-quoc",   # Di chuyển quanh đảo
]

# -------------------------------------------------------
# HÀM CHÍNH: scrape 1 URL bằng Jina Reader
# -------------------------------------------------------

def scrape_one_url(url: str) -> str:
    """
    Nhận vào 1 URL bình thường
    Trả về nội dung chữ sạch từ trang đó

    Ví dụ:
      Input:  "https://visitphuquoc.com.vn"
      Output: "Phú Quốc là hòn đảo... blah blah..."
    """
    jina_url = f"https://r.jina.ai/{url}"
    # Thêm r.jina.ai/ vào trước URL -> Jina sẽ scrape và trả về text sạch

    try:
        response  = requests.get(
            jina_url,
            headers={"Accept": "text/plain"},   # yêu cầu trả về text thuần, không phải HTML
            timeout=30                          # chờ tối đa 30 giây, nếu quá thì bỏ qua
        )

        if response.status_code == 200:
            # 200 = thành công 
            print(f"✅ Lấy được: {url}")
            return response.text
        else:
            # Ví dụ 404 = không tìm thấy trang, 500 = lỗi server
            print(f"⚠️ Lỗi {response.status_code}: {url}")
            return ""
        
    except Exception as e:
        # Lỗi mạng, timeout, v.v.
        print (f"❌ Không kết nối được: {url} | Lỗi: {e}")
        return ""
    
# -------------------------------------------------------
# HÀM KIỂM TRA: file có cũ hơn X ngày không?
# -------------------------------------------------------
def is_data_outdated(filepath: str, days: int = 7) -> bool:
    """
     Kiểm tra xem file dữ liệu có cần cập nhật không.

    Trả về True  → cần scrape lại (file cũ hoặc chưa có)
    Trả về False → còn dùng được (file mới)

    Ví dụ: is_data_outdated("data/DuLich.txt", days=7)
      - Nếu file chưa tồn tại       → True  (cần tạo mới)
      - Nếu file được tạo 10 ngày trước → True  (quá 7 ngày, cần cập nhật)
      - Nếu file được tạo 3 ngày trước  → False (còn mới, dùng tiếp)
    """

    # Kiểm tra file có tồn tại chưa
    if not os.path.exists(filepath):
        print(f" 📂 Chưa có file {filepath} → cần tạo mới")
        return True # chưa có file -> cần scrape

    # ✅ THÊM MỚI: kiểm tra file có bị rỗng không
    if os.path.getsize(filepath) == 0:
        print(f"  📂 File {filepath} đang rỗng → cần scrape lại")
        return True  # ← file rỗng = coi như chưa có dữ liệu

    # Lấy thời gian chỉnh sửa cuối cùng của file
    last_modified_timestamp = os.path.getmtime(filepath)
    # getmtime trả về số giây từ 1970 đến lúc file được sửa lần cuối

    last_modified = datetime.fromtimestamp(last_modified_timestamp)
    # Chuyển số giây đó thành dạng ngày giờ bình thường

    now = datetime.now()
    age = now - last_modified
    # age = khoảng thời gian từ lúc sửa file đến hiện tại

    print(f" 📅 File {filepath} được tạo {age.days} ngày trước")

    return age > timedelta(days=days)
    # timedelta (days=7) = 7 ngày
    # Nếu age > 7 ngày -> True (cũ) | Nếu age <= 7 ngày -> False (mới)



# -------------------------------------------------------
# HÀM TỔNG: scrape tất cả và lưu vào file
# -------------------------------------------------------
def scrape_and_save(filepath: str = "data/DuLich.txt"):
    """
    Scrape tất cả URL trong danh sách URLS
    Ghép lại thành 1 đoạn text lớn
    Lưu vào file filepath
    """
    print("🌐 Bắt đầu scrape dữ liệu du lịch...")

    all_text = "" # biến chứa toàn bộ nội dung scrape được

    for url in URLS:
        print(f" 📡 Đang lấy: {url}")

        content = scrape_one_url(url) # gọi hàm scrape 1 URL

        if content: # nếu lấy được nội dung (không rỗng)
            all_text += f"\n\n=== Nguồn: {url} ===\n\n"
            # Thêm tiêu đề để biết đoạn text này từ trang nào
            all_text += clean_text(content)

        time.sleep(1)
        # Dừng 1 giây giữa mỗi request
        # Lý do: robots.txt của visitphuquoc.com.vn yêu cầu Crawl-Deplay: 1

    # Lưu vào file
    os.makedirs("data", exist_ok=True)
    # exist_ok=True: nếu thư mục data/ chưa có thì tạo mới, có rồi thì thôi
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"✅ Đã lưu {len(all_text)} ký tự vào {filepath}")
    return all_text
    
