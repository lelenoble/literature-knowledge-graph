import re
import collections
import fitz
from services.nlp_service import detect_language, split_sections, segment_text


def parse_pdf(file_bytes: bytes, filename: str) -> dict:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return {"success": False, "error": f"PDF 无法打开: {str(e)}"}

    if doc.is_encrypted:
        doc.close()
        return {"success": False, "error": "加密的 PDF，无法读取"}

    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip()
    author_raw = (meta.get("author") or "").strip()

    full_text_parts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text_parts.append(text)

    doc.close()
    full_text = "\n".join(full_text_parts)

    if len(full_text.strip()) < 100:
        return {"success": False, "error": "PDF 文本内容不足（可能是扫描版图片 PDF）"}

    if not title:
        title = filename.rsplit(".", 1)[0] if "." in filename else filename

    authors = _parse_authors(author_raw)

    year = _extract_year(full_text, meta)

    language = detect_language(full_text)

    # 章节切分只用前 80K 字符，足够找到所有章节标题
    sections = split_sections(full_text[:80000], language)
    abstract = sections.get("abstract", full_text[:1500])
    discussion = sections.get("discussion", "")

    # NLP 只用前 15000 字符 + Counter 词频（比 jieba TF-IDF 快 10x+）
    nlp_text = full_text[:15000]
    words, _ = segment_text(nlp_text, set())
    word_counts = collections.Counter(words)
    keywords = [w for w, _ in word_counts.most_common(20)]

    return {
        "success": True,
        "filename": filename,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract[:2000],
        "full_text": full_text,
        "keywords": keywords,
        "sections": {
            "abstract": abstract[:3000],
            "discussion": discussion[:5000],
        },
        "language": language,
    }


def _parse_authors(raw: str) -> list[str]:
    if not raw:
        return []
    raw = raw.replace("; ", ";").replace(", ", ",")
    parts = re.split(r'[;,]', raw)
    result = []
    for p in parts:
        p = p.strip()
        if p and len(p) < 60:
            result.append(p)
    return result[:10]


def _extract_year(full_text: str, meta: dict) -> int | None:
    year = None
    for field in ["creationDate", "modDate"]:
        val = meta.get(field, "")
        m = re.search(r'(\d{4})', val)
        if m:
            y = int(m.group(1))
            if 1900 <= y <= 2030:
                year = y
                break

    if year is None:
        m = re.search(r'(?:20\d{2}|19\d{2})', full_text[:2000])
        if m:
            year = int(m.group(0))

    return year
