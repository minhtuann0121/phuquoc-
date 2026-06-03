import re

# -------------------------------------------------------
# BƯỚC 1: LÀM SẠCH DỮ LIỆU THÔ
# -------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Làm sạch văn bản thô scrape từ web.

    Loại bỏ:
    - Dòng quá ngắn (menu, nút bấm, nhãn UI)
    - URL thuần
    - Dòng copyright / privacy
    - Dòng "Xem thêm", "Đọc thêm"
    - Khoảng trắng và dòng trống thừa

    Input:  chuỗi văn bản thô
    Output: chuỗi văn bản đã làm sạch
    """
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()

        # Bỏ dòng rỗng hoặc quá ngắn (< 15 ký tự = menu, label, v.v.)
        if len(line) < 15:
            continue

        # Bỏ dòng chỉ là URL
        if re.match(r'^https?://', line):
            continue

        # Bỏ dòng copyright / legal
        if re.search(
            r'copyright|all rights reserved|©|\bprivacy\b|\bterms\b|\bcookies?\b',
            line, re.IGNORECASE
        ):
            continue

        # Bỏ dòng CTA phổ biến
        if re.match(
            r'^(xem thêm|đọc thêm|tìm hiểu thêm|click here|read more|learn more|see more)',
            line, re.IGNORECASE
        ):
            continue

        # Bỏ dòng chỉ toàn ký tự đặc biệt / số / dấu phân cách
        if re.match(r'^[\W\d_]+$', line):
            continue

        clean_lines.append(line)

    # Ghép lại và thu gọn khoảng trắng thừa
    result = "\n".join(clean_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)   # tối đa 2 dòng trống liên tiếp
    result = re.sub(r'[ \t]{2,}', ' ', result)   # xoá khoảng trắng đôi trong dòng

    return result.strip()


# -------------------------------------------------------
# BƯỚC 2: CHUNKING (CẮT THÀNH ĐOẠN NHỎ)
# -------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    source_url: str = ""
) -> list[dict]:
    """
    Cắt văn bản thành các đoạn (chunks) nhỏ để embed vào ChromaDB.

    Tham số:
    - text       : văn bản đầu vào (đã clean)
    - chunk_size : số từ tối đa trong 1 chunk (mặc định 300 từ)
    - overlap    : số từ dùng chung giữa 2 chunk liền kề (mặc định 50 từ)
                   → tránh mất ngữ cảnh ở ranh giới chunk
    - source_url : URL nguồn, lưu vào metadata của chunk

    Trả về: danh sách dict, mỗi dict gồm:
    {
        "id"      : "chunk_0", "chunk_1", ...
        "text"    : nội dung đoạn văn
        "metadata": { "source": url, "chunk_index": 0, "word_count": 280 }
    }

    Ví dụ:
    >>> chunks = chunk_text("Phú Quốc là hòn đảo...", chunk_size=300, overlap=50)
    >>> print(chunks[0]["text"])
    'Phú Quốc là hòn đảo...'
    """

    # Tách thành danh sách từ
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)

        # Lấy đoạn từ start đến end
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "id": f"chunk_{chunk_index}",
            "text": chunk_text_str,
            "metadata": {
                "source": source_url,
                "chunk_index": chunk_index,
                "word_count": len(chunk_words),
            }
        })

        chunk_index += 1

        # Bước nhảy = chunk_size - overlap
        # Ví dụ: chunk_size=300, overlap=50 → bước nhảy=250
        # Chunk kế tiếp bắt đầu từ từ thứ 250, chia sẻ 50 từ cuối với chunk trước
        step = chunk_size - overlap
        start += step

        # Nếu phần còn lại < overlap thì không tạo chunk mới (tránh chunk quá nhỏ)
        if total_words - start < overlap:
            break

    return chunks


# -------------------------------------------------------
# HÀM TỔNG: CLEAN + CHUNK TOÀN BỘ FILE
# -------------------------------------------------------

def process_raw_file(
    filepath: str = "data/DuLich.txt",
    chunk_size: int = 300,
    overlap: int = 50
) -> list[dict]:
    """
    Đọc file DuLich.txt, tách theo từng section (=== Nguồn: ... ===),
    clean từng section rồi chunk.

    Trả về toàn bộ danh sách chunks từ tất cả các section.

    Cách dùng trong main.py:
    >>> from chunker import process_raw_file
    >>> all_chunks = process_raw_file()
    >>> print(f"Tổng số chunks: {len(all_chunks)}")
    """

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Tách theo marker "=== Nguồn: <url> ==="
    # Pattern: dòng bắt đầu bằng === Nguồn:
    sections = re.split(r'={3}\s*Nguồn:\s*(https?://\S+)\s*={3}', raw)
    # sections = ["phần trước marker", "url1", "nội dung1", "url2", "nội dung2", ...]

    all_chunks = []

    # Duyệt từng cặp (url, nội dung)
    # sections[0] là phần trước marker đầu tiên (thường rỗng), bỏ qua
    i = 1
    while i + 1 < len(sections):
        source_url = sections[i].strip()
        content = sections[i + 1].strip()

        # Làm sạch
        cleaned = clean_text(content)

        if not cleaned:
            i += 2
            continue

        # Chunk
        chunks = chunk_text(
            text=cleaned,
            chunk_size=chunk_size,
            overlap=overlap,
            source_url=source_url
        )

        # Đặt lại ID để không trùng giữa các section
        for chunk in chunks:
            chunk["id"] = f"{source_url.split('/')[-1]}_{chunk['id']}"

        all_chunks.extend(chunks)
        i += 2

    print(f"✅ Tổng: {len(all_chunks)} chunks từ {filepath}")
    return all_chunks


# -------------------------------------------------------
# CHẠY THỬ (test nhanh)
# -------------------------------------------------------

if __name__ == "__main__":
    sample = """
    === Nguồn: https://visitphuquoc.com.vn/vi/diem-den ===

    Thị trấn Hoàng Hôn là điểm đến nổi tiếng của Phú Quốc, nơi du khách có thể
    ngắm hoàng hôn tuyệt đẹp mỗi buổi chiều. Đây cũng là trung tâm mua sắm,
    ẩm thực và giải trí sầm uất nhất trên đảo.

    Bãi Kem nằm ở phía Nam Phú Quốc, được vinh danh top 50 bãi biển đẹp nhất
    hành tinh. Bãi cát trắng mịn kéo dài với làn nước trong xanh là thiên đường
    dành cho những ai yêu biển.

    Xem thêm

    https://visitphuquoc.com.vn/something-else

    © 2024 Visit Phu Quoc. All rights reserved.
    """

    # Ghi ra file tạm để test
    import os
    os.makedirs("data", exist_ok=True)
    with open("data/DuLich.txt", "w", encoding="utf-8") as f:
        f.write(sample)

    chunks = process_raw_file(chunk_size=50, overlap=10)
    for c in chunks:
        print(f"\n--- {c['id']} ({c['metadata']['word_count']} từ) ---")
        print(c["text"][:120], "...")