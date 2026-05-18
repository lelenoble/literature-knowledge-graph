import re
import collections
import jieba
import jieba.analyse
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def segment_text(
    content: str,
    custom_stopwords: set[str] | None = None,
    allowed_pos: set[str] | None = None,
    use_pos: bool = False,
) -> tuple[list[str], str]:
    if custom_stopwords is None:
        custom_stopwords = set()
    if allowed_pos is None:
        allowed_pos = {"n", "a", "nr", "ns", "nt", "nz"}

    if use_pos:
        words = []
        for item in pseg.lcut(content):
            if len(item.word) > 1 and item.word not in custom_stopwords and item.flag in allowed_pos:
                words.append(item.word)
        if len(words) < 10:
            words = []
            for item in pseg.lcut(content):
                if len(item.word) > 1 and item.word not in custom_stopwords:
                    words.append(item.word)
    else:
        words = [w for w in jieba.lcut(content) if len(w) > 1 and w not in custom_stopwords]

    processed_text = " ".join(words)
    return words, processed_text


def extract_tfidf(processed_text: str, top_k: int = 50) -> list[tuple[str, float]]:
    results = jieba.analyse.extract_tags(processed_text, topK=max(top_k, 5000), withWeight=True)
    return results[:top_k]


def build_cooccurrence(
    content: str,
    top_words: list[str],
    min_count: int = 2,
) -> tuple[list[dict], list[dict]]:
    sentences = re.split(r'[。！？.!?\n]+', content)
    co_matrix = collections.defaultdict(int)

    for sentence in sentences:
        if not sentence.strip():
            continue
        sen_words = [w for w in jieba.lcut(sentence) if w in top_words]
        sen_words = list(set(sen_words))
        for i in range(len(sen_words)):
            for j in range(i + 1, len(sen_words)):
                w1, w2 = sorted([sen_words[i], sen_words[j]])
                co_matrix[(w1, w2)] += 1

    tfidf_dict = {}
    nodes = []
    for i, w in enumerate(top_words):
        nodes.append({"name": w, "symbolSize": max(10, (i < 10) * 30 + 10)})

    links = []
    for (w1, w2), count in co_matrix.items():
        if count >= min_count:
            links.append({"source": w1, "target": w2, "value": count})

    links.sort(key=lambda x: x["value"], reverse=True)
    return nodes, links[:40]


def build_semantic_clusters(
    content: str,
    top_words: list[str],
    tfidf_dict: dict[str, float],
) -> dict:
    sentences = re.split(r'[。！？.!?\n]+', content)
    processed_sentences = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        sen_words = [w for w in jieba.lcut(sentence) if w in top_words]
        processed_sentences.append(" ".join(sen_words))

    tree_data = {"name": "全局语义", "children": []}
    if len(processed_sentences) > 5 and len(top_words) >= 5:
        try:
            vectorizer = TfidfVectorizer(vocabulary=top_words)
            tfidf_matrix = vectorizer.fit_transform(processed_sentences).T

            num_clusters = min(4, len(top_words) // 4)
            num_clusters = max(2, num_clusters)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix.toarray())

            clusters = collections.defaultdict(list)
            for word, label in zip(top_words, kmeans.labels_):
                clusters[label].append((word, tfidf_dict.get(word, 0)))

            for label, words_in_cluster in clusters.items():
                words_in_cluster.sort(key=lambda x: x[1], reverse=True)
                cluster_core = words_in_cluster[0][0]
                tree_data["children"].append({
                    "name": f"核心: {cluster_core}",
                    "children": [{"name": w, "value": round(wt, 4)} for w, wt in words_in_cluster],
                })
        except Exception:
            pass

    return tree_data


def detect_language(text: str) -> str:
    cjk_count = sum(1 for ch in text if '一' <= ch <= '鿿')
    total = len(text)
    if total == 0:
        return "zh"
    return "zh" if cjk_count / total > 0.05 else "en"


def split_sections(full_text: str, language: str = "zh") -> dict[str, str]:
    if language == "zh":
        patterns = {
            "abstract": r'(?:摘\s*要|ABSTRACT|Abstract)[：:\s]*',
            "introduction": r'(?:引\s*言|前\s*言|绪\s*论|INTRODUCTION|Introduction)[：:\s]*',
            "method": r'(?:方\s*法|实\s*验|METHOD|Method|Experiment)[：:\s]*',
            "discussion": r'(?:讨\s*论|分\s*析|结\s*果|DISCUSSION|Discussion|RESULT|Result)[：:\s]*',
            "references": r'(?:参\s*考\s*文\s*献|REFERENCES|References|Bibliography)[：:\s]*',
        }
    else:
        patterns = {
            "abstract": r'(?:ABSTRACT|Abstract)[：:\s]*',
            "introduction": r'(?:INTRODUCTION|Introduction)[：:\s]*',
            "method": r'(?:METHOD|Method|Experiment)[：:\s]*',
            "discussion": r'(?:DISCUSSION|Discussion|RESULT|Result)[：:\s]*',
            "references": r'(?:REFERENCES|References|Bibliography)[：:\s]*',
        }

    sections = {}
    last_pos = 0
    last_key = "preamble"

    matches = []
    for key, pattern in patterns.items():
        for m in re.finditer(pattern, full_text):
            matches.append((m.start(), key))

    matches.sort()

    for pos, key in matches:
        if pos < last_pos:
            continue
        if last_key not in sections:
            sections[last_key] = full_text[last_pos:pos].strip()
        else:
            sections[last_key] += "\n" + full_text[last_pos:pos].strip()
        last_key = key
        last_pos = pos

    sections[last_key] = full_text[last_pos:].strip()

    if "preamble" in sections and len(sections) > 1:
        if "abstract" not in sections:
            sections["abstract"] = sections.pop("preamble")[:2000]

    return sections


def extract_keywords_tfidf(words: list[str], top_k: int = 20) -> list[str]:
    text = " ".join(words)
    results = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
    return results
