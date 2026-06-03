import re

def clean_text(text: str) -> str:
    # 1. Xoá các dòng chỉ có ký tự đặc biệt, số, hoặc rỗng
    lines = text.split("\n")
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Bỏ dòng quá ngắn (menu, nút bấm, v.v.)
        if len(line) < 15:
            continue
        
        # Bỏ dòng chứa toàn URL
        if re.match(r'^https?://', line):
            continue
        
        # Bỏ dòng kiểu "Copyright © 2024", "All rights reserved"
        if re.search(r'copyright|all rights reserved|©|\bprivacy\b', line, re.IGNORECASE):
            continue
        
        # Bỏ các dòng lặp lại như "Xem thêm | Đọc thêm | ..."
        if re.match(r'^(xem thêm|đọc thêm|tìm hiểu thêm|click here)', line, re.IGNORECASE):
            continue
        
        clean_lines.append(line)
    
    # 2. Ghép lại, thu gọn khoảng trắng thừa
    result = "\n".join(clean_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)  # tối đa 2 dòng trống liên tiếp
    result = re.sub(r' {2,}', ' ', result)       # xoá khoảng trắng đôi
    
    return result.strip()