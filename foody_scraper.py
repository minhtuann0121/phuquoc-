"""
foody_scraper.py
================
Scrape dữ liệu quán ăn Phú Quốc từ Foody.vn qua Jina Reader.
Lưu ra DuLich.txt theo đúng format chunker.py đang dùng.

Pipeline:
  1. Scrape trang danh sách  → lấy URL từng quán
  2. Scrape trang chi tiết   → lấy tên, mô tả, địa chỉ, tag
  3. Clean text              → dùng cleaner.py hiện có
  4. Lưu vào DuLich.txt      → chunker.py đọc được ngay

Cách chạy:
    python foody_scraper.py
"""

import requests
import time
import re
import os
from cleaner import clean_text

# -------------------------------------------------------
# CẤU HÌNH
# -------------------------------------------------------

# Jina Reader endpoint — thêm vào trước bất kỳ URL nào
JINA_PREFIX = "https://r.jina.ai/"

# Các trang danh sách quán ăn Phú Quốc trên Foody
# Foody phân loại theo slug, mỗi trang ~20 quán, có pagination
LIST_URLS = [
    "https://www.foody.vn/phu-quoc/quan-an",
    "https://www.foody.vn/phu-quoc/nha-hang",
    "https://www.foody.vn/phu-quoc/hai-san",
    "https://www.foody.vn/phu-quoc/cafe",
    "https://www.foody.vn/phu-quoc/an-vat",
]

# Số trang pagination tối đa mỗi category (tránh scrape quá nhiều)
MAX_PAGES = 3  # page=1 đến page=3 → ~60 quán/category

# File output — đúng format chunker.py
OUTPUT_FILE = "data/DuLich.txt"

# Delay giữa các request (giây) — tránh bị block
DELAY = 1.5


# -------------------------------------------------------
# BƯỚC 1: GỌI JINA READER
# -------------------------------------------------------

def jina_get(url: str, timeout: int = 30) -> str:
    """
    Dùng Jina Reader (r.jina.ai) để fetch 1 URL, trả về text thuần.
    Jina tự xử lý JavaScript rendering — không cần Selenium.

    Trả về chuỗi rỗng nếu thất bại.
    """
    jina_url = JINA_PREFIX + url
    try:
        resp = requests.get(
            jina_url,
            headers={
                "Accept": "text/plain",
                "User-Agent": "Mozilla/5.0 (compatible; RAG-bot/1.0)"
            },
            timeout=timeout
        )
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"  ⚠️  HTTP {resp.status_code}: {url}")
            return ""
    except Exception as e:
        print(f"  ❌ Lỗi kết nối: {e}")
        return ""


# -------------------------------------------------------
# BƯỚC 2: TRÍCH XUẤT URL QUÁN TỪ TRANG DANH SÁCH
# -------------------------------------------------------

def extract_restaurant_urls(list_page_text: str, base_domain: str = "https://www.foody.vn") -> list[str]:
    """
    Từ text của trang danh sách (Jina đã render), tìm các URL quán cụ thể.

    Foody có pattern URL:
    - /phu-quoc/quan-an/ten-quan-abc-xyz
    - /phu-quoc/nha-hang/ten-nha-hang

    Jina trả về Markdown, links có dạng:
    [Tên quán](https://www.foody.vn/phu-quoc/quan-an/ten-quan)
    """
    urls = set()

    # Pattern 1: Markdown links từ Jina
    # [text](https://www.foody.vn/phu-quoc/.../ten-quan)
    md_links = re.findall(
        r'\(https://www\.foody\.vn/(phu-quoc/[^/]+/[^)\s]+)\)',
        list_page_text
    )
    for path in md_links:
        # Bỏ qua các path không phải trang quán (category page, search...)
        parts = path.split("/")
        if len(parts) == 3:  # phu-quoc / category / slug-quan
            urls.add(f"https://www.foody.vn/{path}")

    # Pattern 2: URL thuần trong text
    raw_links = re.findall(
        r'https://www\.foody\.vn/phu-quoc/[^/]+/[\w-]+',
        list_page_text
    )
    for link in raw_links:
        parts = link.replace("https://www.foody.vn/", "").split("/")
        if len(parts) == 3:
            urls.add(link)

    return list(urls)


# -------------------------------------------------------
# BƯỚC 3: PARSE TRANG CHI TIẾT QUÁN
# -------------------------------------------------------

def parse_restaurant_detail(text: str, source_url: str) -> str:
    """
    Từ text Jina của trang chi tiết 1 quán, trích xuất thông tin
    và trả về đoạn text sạch dạng văn xuôi để lưu vào DuLich.txt.

    Jina trả về Markdown — mình parse các phần quan trọng:
    - Tên quán (thường là heading # hoặc ## đầu tiên)
    - Địa chỉ (dòng có "Địa chỉ:" hoặc pattern địa chỉ VN)
    - Mô tả / giới thiệu (đoạn văn dài nhất)
    - Tags / món ăn nổi bật
    - Giờ mở cửa
    - Giá trung bình
    """

    lines = text.split("\n")
    result_parts = []

    ten_quan = ""
    dia_chi = ""
    gio_mo_cua = ""
    gia = ""
    mo_ta_lines = []
    tags = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Tên quán — heading đầu tiên
        if not ten_quan and re.match(r'^#{1,3}\s+\S', line):
            ten_quan = re.sub(r'^#+\s+', '', line).strip()
            i += 1
            continue

        # Địa chỉ
        if re.search(r'(địa chỉ|address|Đường|đường|phường|Phường|quận|Quận|huyện)', line, re.IGNORECASE):
            # Bỏ label, giữ nội dung
            addr = re.sub(r'^(địa chỉ|address)\s*[:\-]\s*', '', line, flags=re.IGNORECASE).strip()
            if addr and len(addr) > 5:
                dia_chi = addr

        # Giờ mở cửa
        if re.search(r'(giờ|mở cửa|opening|giờ làm việc)', line, re.IGNORECASE):
            gio = re.sub(r'^(giờ|mở cửa|opening hours?|giờ làm việc)\s*[:\-]\s*', '', line, flags=re.IGNORECASE).strip()
            if gio and len(gio) > 3:
                gio_mo_cua = gio

        # Giá
        if re.search(r'(giá|price|khoảng giá|mức giá)', line, re.IGNORECASE):
            g = re.sub(r'^(giá|price|khoảng giá|mức giá)\s*[:\-]\s*', '', line, flags=re.IGNORECASE).strip()
            if g and len(g) > 2:
                gia = g

        # Tags / hashtag / danh mục
        if re.match(r'^[#\*]\s*[\w\s]+$', line) or re.match(r'^\[.*\]\(.*\)$', line):
            tag = re.sub(r'^[#\*\[\]()]+\s*', '', line).strip()
            if 2 < len(tag) < 40:
                tags.append(tag)

        # Đoạn văn mô tả: dòng dài (>50 ký tự), không phải heading, không phải URL
        if (
            len(line) > 50
            and not line.startswith('#')
            and not line.startswith('http')
            and not line.startswith('|')  # bỏ bảng Markdown
            and not re.match(r'^\[.*\]\(http', line)  # bỏ Markdown links
        ):
            mo_ta_lines.append(line)

        i += 1

    # --- Ghép thành văn bản ---

    if ten_quan:
        result_parts.append(f"Tên quán: {ten_quan}")

    if dia_chi:
        result_parts.append(f"Địa chỉ: {dia_chi}")

    if gio_mo_cua:
        result_parts.append(f"Giờ mở cửa: {gio_mo_cua}")

    if gia:
        result_parts.append(f"Giá trung bình: {gia}")

    if tags:
        result_parts.append(f"Loại hình: {', '.join(tags[:8])}")

    # Mô tả: lấy tối đa 10 dòng dài nhất (loại bỏ trùng lặp)
    seen = set()
    mo_ta_unique = []
    for line in mo_ta_lines:
        normalized = re.sub(r'\s+', ' ', line).strip()
        if normalized not in seen:
            seen.add(normalized)
            mo_ta_unique.append(normalized)

    if mo_ta_unique:
        result_parts.append("\nMô tả:")
        result_parts.extend(mo_ta_unique[:10])

    return "\n".join(result_parts)


# -------------------------------------------------------
# BƯỚC 4: SCRAPE TOÀN BỘ — DANH SÁCH → CHI TIẾT → LƯU FILE
# -------------------------------------------------------

def scrape_foody(
    list_urls: list = LIST_URLS,
    max_pages: int = MAX_PAGES,
    output_file: str = OUTPUT_FILE,
    max_detail_per_category: int = 20,
) -> str:
    """
    Hàm chính: scrape Foody và lưu ra DuLich.txt.

    Pipeline:
        list_urls → pagination → extract restaurant URLs
                 → scrape từng trang chi tiết
                 → parse → clean → lưu file

    Tham số:
        max_pages              : số trang pagination mỗi category
        max_detail_per_category: tối đa bao nhiêu quán lấy chi tiết
                                 (tránh scrape quá lâu)
    """
    print("=" * 60)
    print("🍜 Foody Scraper — Quán ăn Phú Quốc")
    print("=" * 60)

    all_text = ""
    all_restaurant_urls = set()

    # ── GIAI ĐOẠN 1: Thu thập URL quán từ trang danh sách ──
    print("\n📋 Giai đoạn 1: Lấy danh sách quán...")

    for base_url in list_urls:
        category = base_url.split("/")[-1]
        print(f"\n  📂 Category: {category}")

        for page in range(1, max_pages + 1):
            # Foody dùng ?page=N hoặc /page/N
            if page == 1:
                url = base_url
            else:
                url = f"{base_url}?page={page}"

            print(f"    📡 Trang {page}: {url}")
            text = jina_get(url)

            if not text:
                print(f"    ⏭️  Không lấy được, bỏ qua")
                break

            # Trích URL quán
            found = extract_restaurant_urls(text)
            new_found = [u for u in found if u not in all_restaurant_urls]
            all_restaurant_urls.update(found)

            print(f"    ✅ Tìm thấy {len(found)} quán ({len(new_found)} mới)")

            # Nếu trang này không có quán nào → đã hết trang
            if not found:
                print(f"    ⏹️  Hết dữ liệu ở trang {page}")
                break

            time.sleep(DELAY)

    print(f"\n✅ Tổng URL quán tìm được: {len(all_restaurant_urls)}")

    # ── GIAI ĐOẠN 2: Scrape chi tiết từng quán ──
    print("\n📖 Giai đoạn 2: Lấy chi tiết từng quán...")

    # Giới hạn tổng số quán để tránh scrape quá lâu
    restaurant_urls = list(all_restaurant_urls)[:max_detail_per_category * len(list_urls)]
    total = len(restaurant_urls)

    success = 0
    for idx, url in enumerate(restaurant_urls, 1):
        print(f"  [{idx}/{total}] {url}")

        text = jina_get(url)
        if not text:
            continue

        # Parse chi tiết
        detail_text = parse_restaurant_detail(text, source_url=url)

        # Clean
        cleaned = clean_text(detail_text)

        if len(cleaned) < 50:
            print(f"    ⚠️  Nội dung quá ngắn, bỏ qua")
            continue

        # Thêm vào file theo format chunker.py
        # === Nguồn: <url> === là marker mà chunker.py dùng để tách section
        all_text += f"\n\n=== Nguồn: {url} ===\n\n"
        all_text += cleaned

        success += 1
        print(f"    ✅ OK ({len(cleaned)} ký tự)")

        time.sleep(DELAY)

    # ── GIAI ĐOẠN 3: Lưu file ──
    print(f"\n💾 Giai đoạn 3: Lưu file...")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"\n{'='*60}")
    print(f"✅ Hoàn thành!")
    print(f"   Quán scrape thành công : {success}/{total}")
    print(f"   Tổng ký tự             : {len(all_text)}")
    print(f"   File output            : {output_file}")
    print(f"{'='*60}")

    return all_text


# -------------------------------------------------------
# TÍCH HỢP VÀO SCRAPER.PY HIỆN CÓ
# -------------------------------------------------------
# Trong scraper.py, thay hàm scrape_and_save() bằng:
#
#   from foody_scraper import scrape_foody
#
#   def scrape_and_save(filepath="data/DuLich.txt"):
#       return scrape_foody(output_file=filepath)
#
# Hoặc gọi trực tiếp trong main.py:
#
#   from foody_scraper import scrape_foody
#   if is_data_outdated("data/DuLich.txt"):
#       scrape_foody()
# -------------------------------------------------------


# -------------------------------------------------------
# CHẠY TRỰC TIẾP
# -------------------------------------------------------

if __name__ == "__main__":
    scrape_foody(
        list_urls=LIST_URLS,
        max_pages=MAX_PAGES,           # 3 trang × 5 category = ~15 request danh sách
        max_detail_per_category=20,    # tối đa 20 quán/category × 5 = 100 quán
        output_file=OUTPUT_FILE,
    )

    # Kiểm tra nhanh output
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        sections = re.findall(r'=== Nguồn:', content)
        print(f"\n📊 File DuLich.txt: {len(sections)} quán, {len(content)} ký tự")
        print("   Sẵn sàng đưa vào chunker.py ✅")
