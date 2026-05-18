"""
文献知识图谱工具 - 后端服务
Literature Knowledge Graph + NLP Analytics Backend

功能：
- 外层：PDF 批量上传 → 论文知识图谱 → 概念溯源 → 研究缺口检测
- 内层：单文本 NLP 深度分析（词云/TF-IDF/共现网络/语义聚类 + 上下文搭配分析）
"""

import os
import re
import io
import base64
import time
import json
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify
import jieba
import jieba.analyse
import jieba.posseg as pseg
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from storage.paper_store import PaperStore
from models.relation import Relation, RelationType, RELATION_LABELS, RELATION_COLORS
from services.pdf_parser import parse_pdf

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

_stores: dict[str, PaperStore] = {}
_stores_lock = threading.Lock()
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()
_cluster_meta_by_session: dict[str, dict] = {}
_llm_configs: dict[str, dict] = {}

_bg_sid = threading.local()


def _session_id() -> str:
    sid = getattr(_bg_sid, 'value', None)
    if sid:
        return sid
    from flask import session
    if "uid" not in session:
        session["uid"] = os.urandom(16).hex()
    return session["uid"]


def _get_store() -> PaperStore:
    sid = _session_id()
    with _stores_lock:
        if sid not in _stores:
            _stores[sid] = PaperStore()
        return _stores[sid]


def _get_tasks() -> dict:
    sid = _session_id()
    with _tasks_lock:
        if sid not in _tasks:
            _tasks[sid] = {}
        return _tasks[sid]


def _get_cluster_meta() -> dict:
    sid = _session_id()
    if sid not in _cluster_meta_by_session:
        _cluster_meta_by_session[sid] = {}
    return _cluster_meta_by_session[sid]

# ---------------- LLM 配置 ----------------
def _get_llm_config() -> dict:
    sid = _session_id()
    if sid not in _llm_configs:
        _llm_configs[sid] = {
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        }
    return _llm_configs[sid]


# ---------------- 工具函数 ----------------

def get_chinese_font():
    import platform
    system = platform.system()
    if system == 'Darwin':
        paths = [
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            '/System/Library/Fonts/PingFang.ttc',
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    elif system == 'Windows':
        paths = ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']
        for p in paths:
            if os.path.exists(p):
                return p
    import wordcloud.wordcloud
    return wordcloud.wordcloud.FONT_PATH


def _call_llm(messages: list[dict], expect_json: bool = False) -> dict | None:
    """通用 LLM 调用，返回解析后的 JSON 或 None"""
    cfg = _get_llm_config()
    if not cfg["api_key"]:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=15.0)
        kwargs = dict(model=cfg["model"], messages=messages, temperature=0.3)
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content.strip()
        if expect_json:
            # 去掉可能的 markdown 代码块包裹
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(content)
        return {"content": content}
    except json.JSONDecodeError as e:
        return {"error": f"JSON解析失败: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}


def _build_heuristic_relations() -> list[dict]:
    """纯启发式论文关系检测（不依赖 LLM，作为降级方案）"""
    papers = _get_store().list_all()
    if len(papers) < 2:
        return []

    n = len(papers)
    texts = [p.get("abstract", p.get("full_text", ""))[:3000] for p in papers]
    ids = [p["id"] for p in papers]
    id_to_idx = {p["id"]: i for i, p in enumerate(papers)}

    vectorizer = TfidfVectorizer(max_features=500)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    sim_matrix = cosine_similarity(matrix)
    relations = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i][j]
            if sim < 0.15:
                continue

            kw_i = set(papers[i].get("keywords", []))
            kw_j = set(papers[j].get("keywords", []))
            kw_overlap = len(kw_i & kw_j) / max(len(kw_i | kw_j), 1)

            year_i = papers[i].get("year")
            year_j = papers[j].get("year")

            if "method" in (papers[i].get("abstract", "") + papers[j].get("abstract", "")).lower():
                rel_type = "shared_method"
            else:
                rel_type = "shared_topic"

            confidence = round(sim * 0.7 + kw_overlap * 0.3, 2)
            if confidence < 0.25:
                continue

            source, target = (ids[i], ids[j]) if (year_i or 0) <= (year_j or 0) else (ids[j], ids[i])

            relations.append({
                "source_id": source,
                "target_id": target,
                "relation_type": rel_type,
                "confidence": confidence,
                "evidence": f"摘要相似度 {sim:.2f}, 关键词重叠 {kw_overlap:.2f}",
            })

    for paper in papers:
        pid = paper["id"]
        has_relation = any(r["source_id"] == pid or r["target_id"] == pid for r in relations)
        if not has_relation and len(papers) > 1:
            most_sim_idx = None
            most_sim_val = 0
            for i, p2 in enumerate(papers):
                if p2["id"] == pid:
                    continue
                sim = sim_matrix[id_to_idx[pid]][id_to_idx[p2["id"]]]
                if sim > most_sim_val:
                    most_sim_val = sim
                    most_sim_idx = p2["id"]
            if most_sim_idx and most_sim_val > 0.05:
                relations.append({
                    "source_id": pid,
                    "target_id": most_sim_idx,
                    "relation_type": "shared_topic",
                    "confidence": round(most_sim_val, 2),
                    "evidence": "补充关系，确保每篇论文有至少一个关联",
                })

    return relations


# ---------------- 页面路由 ----------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------- 论文管理 API ----------------

@app.route('/api/papers/upload', methods=['POST'])
def upload_papers():
    if 'files' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    files = request.files.getlist('files')
    results = []

    for file in files:
        if file.filename == '':
            continue
        raw_bytes = file.read()
        result = parse_pdf(raw_bytes, file.filename)
        if result.get("success"):
            paper_id = _get_store().add(result)
            results.append({"id": paper_id, "title": result["title"], "status": "success"})
        else:
            results.append({"filename": file.filename, "status": "error", "error": result.get("error", "未知错误")})

    return jsonify({"papers": results, "total": _get_store().paper_count})


@app.route('/api/papers/list', methods=['GET'])
def list_papers():
    papers = _get_store().list_all()
    return jsonify({
        "papers": [
            {
                "id": p["id"],
                "title": p["title"],
                "authors": p["authors"],
                "year": p["year"],
                "keywords": p["keywords"],
                "language": p["language"],
                "upload_time": p["upload_time"],
            }
            for p in papers
        ]
    })


@app.route('/api/papers/<paper_id>', methods=['GET'])
def get_paper(paper_id):
    paper = _get_store().get(paper_id)
    if not paper:
        return jsonify({'error': '论文不存在'}), 404
    return jsonify({"paper": paper})


@app.route('/api/papers/clear', methods=['DELETE'])
def clear_papers():
    _get_store().clear()
    return jsonify({"success": True, "total": 0})


# ---------------- LLM 配置 API ----------------

@app.route('/api/llm/configure', methods=['POST'])
def configure_llm():
    data = request.get_json() or {}
    if "api_key" in data:
        _get_llm_config()["api_key"] = data["api_key"]
    if "base_url" in data:
        _get_llm_config()["base_url"] = data["base_url"]
    if "model" in data:
        _get_llm_config()["model"] = data["model"]
    return jsonify({"success": True, "configured": bool(_get_llm_config()["api_key"])})


# ---------------- 知识图谱 API ----------------

@app.route('/api/graph/build', methods=['POST'])
def build_graph():
    if _get_store().paper_count < 2:
        return jsonify({'error': '至少需要 2 篇论文才能构建图谱'}), 400

    task_id = f"graph_{int(time.time())}"
    with _tasks_lock:
        _get_tasks()[task_id] = {"status": "running", "progress": 0, "result": None}

    sid = _session_id()

    def _run():
        _get_store().clear_relations()

        has_llm = bool(_get_llm_config()["api_key"])
        if has_llm:
            papers = _get_store().list_all()
            texts = [p.get("abstract", p.get("full_text", ""))[:3000] for p in papers]

            vectorizer = TfidfVectorizer(max_features=500)
            try:
                matrix = vectorizer.fit_transform(texts)
            except ValueError:
                matrix = None

            total_candidates = 0
            classified = 0
            llm_failures = 0
            llm_disabled = False

            if matrix is not None:
                sim_matrix = cosine_similarity(matrix)
                n = len(papers)
                for i in range(n):
                    sims = [(j, sim_matrix[i][j]) for j in range(n) if j != i]
                    sims.sort(key=lambda x: x[1], reverse=True)
                    top_k = sims[:5]
                    for j, sim in top_k:
                        if sim < 0.1:
                            continue
                        total_candidates += 1

                        kw_i = set(papers[i].get("keywords", []))
                        kw_j = set(papers[j].get("keywords", []))
                        kw_overlap = len(kw_i & kw_j) / max(len(kw_i | kw_j), 1)

                        year_i = papers[i].get("year")
                        year_j = papers[j].get("year")

                        if kw_overlap > 0.6 and sim > 0.7:
                            source = papers[i]["id"] if (year_i or 0) <= (year_j or 0) else papers[j]["id"]
                            target = papers[j]["id"] if source == papers[i]["id"] else papers[i]["id"]
                            _get_store().add_relation(Relation(source_id=source, target_id=target,
                                                       relation_type=RelationType.SHARED_TOPIC,
                                                       confidence=round(sim, 2),
                                                       evidence=f"高摘要相似度 {sim:.2f}，关键词重叠 {kw_overlap:.2f}"))
                            classified += 1
                            continue
                        elif kw_overlap > 0.4:
                            source = papers[i]["id"] if (year_i or 0) <= (year_j or 0) else papers[j]["id"]
                            target = papers[j]["id"] if source == papers[i]["id"] else papers[i]["id"]
                            _get_store().add_relation(Relation(source_id=source, target_id=target,
                                                       relation_type=RelationType.SHARED_TOPIC,
                                                       confidence=round(sim, 2),
                                                       evidence=f"关键词重叠 {kw_overlap:.2f}"))
                            classified += 1
                            continue
                        else:
                            # 如果 LLM 已连续失败 3 次，跳过剩余 LLM 调用，最后用启发式补齐
                            if llm_disabled:
                                continue

                            prompt = f"""Analyze the relationship between these two academic papers.
Paper A ({papers[i].get('year', '?')}): {papers[i]['title']}
Abstract: {papers[i].get('abstract', '')[:400]}
Keywords: {', '.join(papers[i].get('keywords', [])[:10])}

Paper B ({papers[j].get('year', '?')}): {papers[j]['title']}
Abstract: {papers[j].get('abstract', '')[:400]}
Keywords: {', '.join(papers[j].get('keywords', [])[:10])}

Choose ONE: cites | improves | refutes | shared_method | shared_topic | none
Return JSON: {{"relation": "...", "confidence": 0.0-1.0, "evidence": "reason"}}"""

                            result = _call_llm([
                                {"role": "system", "content": "You are a research literature analyst. Return only JSON."},
                                {"role": "user", "content": prompt},
                            ], expect_json=True)

                            rel_map = {
                                "cites": RelationType.CITES,
                                "improves": RelationType.IMPROVES,
                                "refutes": RelationType.REFUTES,
                                "shared_method": RelationType.SHARED_METHOD,
                                "shared_topic": RelationType.SHARED_TOPIC,
                            }

                            if result and "relation" in result and result["relation"] != "none":
                                rel_type_enum = rel_map.get(result["relation"], RelationType.SHARED_TOPIC)
                                confidence = float(result.get("confidence", 0.5))
                                if confidence >= 0.3:
                                    source = papers[i]["id"]
                                    target = papers[j]["id"]
                                    _get_store().add_relation(Relation(
                                        source_id=source, target_id=target,
                                        relation_type=rel_type_enum,
                                        confidence=round(confidence, 2),
                                        evidence=result.get("evidence", ""),
                                    ))
                                    classified += 1
                                llm_failures = 0  # 成功后重置失败计数
                            elif result and "error" in result:
                                llm_failures += 1
                                if llm_failures >= 3:
                                    llm_disabled = True
                            else:
                                llm_failures = 0

                        with _tasks_lock:
                            progress = min(90, int((i / n) * 100))
                            _get_tasks()[task_id]["progress"] = progress

            # Fallback: 如果 LLM 分类太少，补充启发式关系
            if classified < _get_store().paper_count:
                heuristic_rels = _build_heuristic_relations()
                for r in heuristic_rels:
                    existing = [
                        rel for rel in _get_store().get_relations()
                        if rel.source_id == r["source_id"] and rel.target_id == r["target_id"]
                    ]
                    if not existing:
                        _get_store().add_relation(Relation(
                            source_id=r["source_id"], target_id=r["target_id"],
                            relation_type=RelationType(r["relation_type"]),
                            confidence=r["confidence"],
                            evidence=r["evidence"],
                        ))
        else:
            heuristic_rels = _build_heuristic_relations()
            for r in heuristic_rels:
                _get_store().add_relation(Relation(
                    source_id=r["source_id"], target_id=r["target_id"],
                    relation_type=RelationType(r["relation_type"]),
                    confidence=r["confidence"],
                    evidence=r["evidence"],
                ))

        # 聚类
        try:
            _run_clustering()
        except Exception:
            import traceback
            traceback.print_exc()

        with _tasks_lock:
            _get_tasks()[task_id] = {"status": "done", "progress": 100, "result": {"relations": _get_store().relation_count, "clusters": len(_get_cluster_meta())}}

    def _run_wrapped():
        _bg_sid.value = sid
        try:
            _run()
        except Exception:
            import traceback
            traceback.print_exc()
            with _tasks_lock:
                _get_tasks()[task_id] = {"status": "error", "progress": 0, "result": {"error": "图谱构建过程出错，请重试"}}
        finally:
            _bg_sid.value = None

    thread = threading.Thread(target=_run_wrapped)
    thread.daemon = True
    thread.start()

    return jsonify({"task_id": task_id, "total_papers": _get_store().paper_count})


# _cluster_meta 现在通过 _get_cluster_meta() 按 session 隔离

CLUSTER_COLORS = [
    '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
]


def _name_clusters_via_llm(clusters_info: list[dict]) -> dict[str, str]:
    """调用 LLM 为每个聚类生成有意义的研究方向名称，返回 {cluster_id: name}"""
    if not _get_llm_config()["api_key"]:
        return {}

    prompt_lines = ["以下是从学术论文中通过 KMeans 聚类得到的若干个研究方向分组。",
                    "请根据每个聚类的论文标题、关键词和摘要，为每个聚类命名一个简洁的研究方向名称（5-12个字）。",
                    '名称应能准确概括该组论文的共同研究主题，避免使用无意义的词如"数据""任务""方法""研究"等泛词。',
                    "请以 JSON 格式返回，格式：{\"names\": {\"cluster_0\": \"研究方向名\", ...}}\n"]
    for c in clusters_info:
        prompt_lines.append(f"=== {c['id']}（{len(c['indices'])}篇论文）===")
        prompt_lines.append(f"论文标题: {'; '.join(c['titles'][:8])}")
        prompt_lines.append(f"高频关键词: {', '.join(c['top_keywords'][:8])}")
        if c.get("abstracts"):
            combined_abstract = " ".join(c["abstracts"])[:800]
            prompt_lines.append(f"摘要摘要: {combined_abstract}")
        prompt_lines.append("")

    prompt = "\n".join(prompt_lines)
    result = _call_llm([
        {"role": "system", "content": "你是一位学术文献分析专家，擅长从论文聚类中识别和命名研究方向。你只返回JSON格式，不使用Markdown代码块。"},
        {"role": "user", "content": prompt},
    ], expect_json=True)

    if result and "names" in result:
        return result["names"]
    if result and "error" not in result:
        return {}
    return {}


def _run_clustering():
    """对所有论文摘要进行 KMeans 聚类，结果写入 store 和 session-scoped cluster_meta"""
    papers = _get_store().list_all()
    n = len(papers)
    if n < 2:
        return

    abstracts = [p.get("abstract", p.get("full_text", ""))[:3000] for p in papers]
    n_clusters = min(8, max(2, n // 5))

    vectorizer = TfidfVectorizer(max_features=500)
    try:
        matrix = vectorizer.fit_transform(abstracts)
    except ValueError:
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix.toarray())

    centroids = kmeans.cluster_centers_

    # 第一遍：收集每个聚类的元数据
    clusters_raw = {}
    for c_id in range(n_clusters):
        indices = [i for i, lb in enumerate(labels) if lb == c_id]
        cluster_paper_ids = []
        cluster_keywords_all = []
        cluster_years = []
        cluster_titles = []
        cluster_abstracts = []

        for idx in indices:
            p = papers[idx]
            p["cluster_id"] = f"cluster_{c_id}"
            cluster_paper_ids.append(p["id"])
            cluster_keywords_all.extend(p.get("keywords", []))
            if p.get("year"):
                cluster_years.append(p["year"])
            cluster_titles.append(p.get("title", ""))
            ab = p.get("abstract", "")
            if ab:
                cluster_abstracts.append(ab[:400])

        kw_counter = collections.Counter(cluster_keywords_all)
        top_kw = [kw for kw, _ in kw_counter.most_common(8)]

        if len(top_kw) < 3:
            title_words = []
            for t in cluster_titles:
                words = jieba.lcut(t)
                title_words.extend([w for w in words if len(w) >= 2
                    and w not in ('基于', '研究', '方法', '应用', '一种', '及其', '中的', '用于', '分析', '模型', '数据', '算法', '系统', '设计', '实现', '采用', '提出', '问题', '不同', '可以', '这个', '我们', '他们', '进行', '一个', '通过', '使用')])
            title_counter = collections.Counter(title_words)
            extra = [w for w, _ in title_counter.most_common(8) if w not in top_kw]
            top_kw = (top_kw + extra)[:8]

        clusters_raw[f"cluster_{c_id}"] = {
            "id": f"cluster_{c_id}",
            "indices": indices,
            "paper_ids": cluster_paper_ids,
            "top_keywords": top_kw,
            "titles": cluster_titles,
            "abstracts": cluster_abstracts,
            "years": cluster_years,
        }

    # 调用 LLM 为聚类命名
    llm_names = _name_clusters_via_llm(list(clusters_raw.values()))

    # 第二遍：构建最终的 cluster_meta
    meta = _get_cluster_meta()
    meta.clear()
    for c_id in range(n_clusters):
        c_key = f"cluster_{c_id}"
        raw = clusters_raw[c_key]

        # 优先用 LLM 命名，降级时用关键词拼接
        if llm_names and llm_names.get(c_key, "").strip():
            label = llm_names[c_key].strip()
        else:
            top_kw = raw["top_keywords"]
            if len(top_kw) >= 2:
                label = " · ".join(top_kw[:3])
            elif top_kw:
                label = top_kw[0]
            else:
                label = f"研究方向 {c_id + 1}"

        cluster_years = raw["years"]
        if len(cluster_years) >= 2:
            year_range = f"{min(cluster_years)}–{max(cluster_years)}"
        elif len(cluster_years) == 1:
            year_range = str(cluster_years[0])
        else:
            year_range = ""

        meta[c_key] = {
            "id": c_key,
            "label": label,
            "top_keywords": raw["top_keywords"],
            "paper_count": len(raw["indices"]),
            "paper_ids": raw["paper_ids"],
            "avg_year": round(sum(cluster_years) / len(cluster_years), 1) if cluster_years else None,
            "year_range": year_range,
            "color": CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)],
            "centroid": centroids[c_id].tolist(),
        }


@app.route('/api/graph/clusters', methods=['GET'])
def get_clusters():
    """返回所有聚类摘要 + 相似度矩阵（气泡图数据）"""
    meta = _get_cluster_meta()
    if not meta:
        return jsonify({"clusters": [], "similarity_matrix": []})

    clusters = []
    for c in meta.values():
        cluster_papers = [_get_store().get(pid) for pid in c["paper_ids"]]
        clusters.append({
            "id": c["id"],
            "label": c["label"],
            "paper_count": c["paper_count"],
            "papers": [{
                "id": p["id"], "title": p["title"],
                "year": p.get("year"), "authors": p.get("authors", [])[:3],
            } for p in cluster_papers if p],
            "top_keywords": c["top_keywords"],
            "avg_year": c["avg_year"],
            "year_range": c.get("year_range", ""),
            "color": c["color"],
        })

    # 聚类间相似度矩阵（基于 centroid cosine similarity）
    cluster_ids = list(meta.keys())
    n = len(cluster_ids)
    sim_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                c_i = meta[cluster_ids[i]]["centroid"]
                c_j = meta[cluster_ids[j]]["centroid"]
                sim = cosine_similarity([c_i], [c_j])[0][0]
                row.append(round(float(sim), 3))
        sim_matrix.append(row)

    return jsonify({"clusters": clusters, "similarity_matrix": sim_matrix})


@app.route('/api/graph/cluster/<cluster_id>', methods=['GET'])
def get_cluster_detail(cluster_id):
    """返回单个聚类的论文列表、内部关系、关键词云"""
    meta = _get_cluster_meta()
    if cluster_id not in meta:
        return jsonify({"error": "聚类不存在"}), 404

    c = meta[cluster_id]
    papers_in_cluster = [_get_store().get(pid) for pid in c["paper_ids"]]
    papers_in_cluster = [p for p in papers_in_cluster if p]
    paper_ids_set = set(c["paper_ids"])

    # 内部关系（source 和 target 都在聚类内）
    all_relations = _get_store().get_relations()
    internal_links = []
    external_links = []
    for r in all_relations:
        link = {
            "source": r.source_id, "target": r.target_id,
            "relation_type": r.relation_type.value,
            "relation_label": RELATION_LABELS.get(r.relation_type, ""),
            "confidence": r.confidence,
            "evidence": r.evidence,
        }
        if r.source_id in paper_ids_set and r.target_id in paper_ids_set:
            internal_links.append(link)
        elif r.source_id in paper_ids_set or r.target_id in paper_ids_set:
            external_links.append(link)

    # 关键词云
    all_kw = []
    for p in papers_in_cluster:
        all_kw.extend(p.get("keywords", []))
    kw_counter = collections.Counter(all_kw)
    max_count = max(kw_counter.values()) if kw_counter else 1
    keyword_cloud = [
        {"word": w, "weight": round(c / max_count, 2)}
        for w, c in kw_counter.most_common(30)
    ]

    return jsonify({
        "cluster": {
            "id": c["id"],
            "label": c["label"],
            "paper_count": c["paper_count"],
            "top_keywords": c["top_keywords"],
            "avg_year": c["avg_year"],
            "color": c["color"],
        },
        "papers": [{
            "id": p["id"], "title": p["title"], "filename": p.get("filename", ""),
            "year": p.get("year"), "authors": p.get("authors", []),
            "keywords": p.get("keywords", []), "abstract": (p.get("abstract", "") or "")[:500],
            "language": p.get("language", "zh"),
        } for p in papers_in_cluster],
        "internal_links": internal_links,
        "external_links": external_links,
        "keyword_cloud": keyword_cloud,
    })


@app.route('/api/graph/data', methods=['GET'])
def get_graph_data():
    papers = _get_store().list_all()

    NODE_COLORS = [
        '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6',
        '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
    ]

    nodes = []
    for i, p in enumerate(papers):
        nodes.append({
            "id": p["id"],
            "name": p["title"][:20] + ("..." if len(p["title"]) > 20 else ""),
            "title": p["title"],
            "authors": ", ".join(p["authors"][:3]) if p["authors"] else "未知",
            "year": p["year"],
            "keywords_count": len(p.get("keywords", [])),
            "symbolSize": 15 + len(p.get("keywords", [])) * 0.8,
            "itemStyle": {"color": NODE_COLORS[i % len(NODE_COLORS)]},
            "category": i % len(NODE_COLORS),
        })

    links = []
    for r in _get_store().get_relations():
        source_node = next((n for n in nodes if n["id"] == r.source_id), None)
        target_node = next((n for n in nodes if n["id"] == r.target_id), None)
        if source_node and target_node:
            links.append({
                "source": r.source_id,
                "target": r.target_id,
                "relation_type": r.relation_type.value,
                "relation_label": RELATION_LABELS.get(r.relation_type, ""),
                "confidence": r.confidence,
                "evidence": r.evidence,
                "lineStyle": {
                    "color": RELATION_COLORS.get(r.relation_type, "#999"),
                    "width": max(1.5, r.confidence * 5),
                    "type": "dashed" if r.relation_type.value == "refutes" else "solid",
                    "curveness": 0.2,
                },
            })

    categories = [{"name": f"聚类 {i+1}", "itemStyle": {"color": NODE_COLORS[i]}} for i in range(min(len(papers), len(NODE_COLORS)))]

    return jsonify({"nodes": nodes, "links": links, "categories": categories})


@app.route('/api/status', methods=['GET'])
def task_status():
    task_id = request.args.get('task_id', '')
    with _tasks_lock:
        task = _get_tasks().get(task_id)
    if not task:
        return jsonify({"status": "not_found"}), 404
    return jsonify(task)


# ---------------- 概念溯源 API ----------------

@app.route('/api/concept/trace', methods=['POST'])
def concept_trace():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({'error': '请输入搜索词'}), 400

    papers = _get_store().list_all()
    timeline = collections.defaultdict(list)
    total_mentions = 0

    for paper in papers:
        full_text = paper.get("full_text", "")
        for m in re.finditer(re.escape(query), full_text):
            start = max(0, m.start() - 30)
            end = min(len(full_text), m.end() + 50)
            context = full_text[start:end].replace("\n", " ")
            year = paper.get("year") or "未知"

            timeline[str(year)].append({
                "paper_id": paper["id"],
                "paper_title": paper["title"],
                "sentence": context.strip(),
                "position": m.start(),
            })
            total_mentions += 1

    timeline_list = sorted(
        [{"year": yr, "papers": entries, "count": len(entries)} for yr, entries in timeline.items()],
        key=lambda x: str(x["year"])
    )

    counts = [t["count"] for t in timeline_list]
    if len(counts) >= 2:
        if counts[-1] > counts[0] * 1.2:
            trend = "increasing"
        elif counts[-1] < counts[0] * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return jsonify({"timeline": timeline_list, "total_mentions": total_mentions, "trend": trend})


# ---------------- AI 摘要 API ----------------

@app.route('/api/summary/paper/<paper_id>', methods=['POST'])
def summarize_paper(paper_id):
    paper = _get_store().get(paper_id)
    if not paper:
        return jsonify({'error': '论文不存在'}), 404

    data = request.get_json() or {}
    style = data.get("style", "brief")

    if not _get_llm_config()["api_key"]:
        return jsonify({"summary": "请先配置 LLM API Key", "key_findings": [], "methodology": "", "limitations": ""})

    length_hint = "2-3 sentences" if style == "brief" else "detailed, 3-4 paragraphs"
    prompt = f"""Summarize this academic paper in {length_hint}. Return JSON:
{{
    "summary": "overall summary",
    "key_findings": ["finding 1", "finding 2"],
    "methodology": "method used",
    "limitations": "limitations mentioned"
}}

Title: {paper['title']}
Authors: {', '.join(paper.get('authors', []))}
Year: {paper.get('year', 'N/A')}
Abstract: {paper.get('abstract', '')[:1500]}
Full text excerpt: {paper.get('full_text', '')[:4000]}"""

    result = _call_llm([
        {"role": "system", "content": "You summarize academic papers. Return only JSON."},
        {"role": "user", "content": prompt},
    ], expect_json=True)

    if result and "error" not in result:
        return jsonify(result)
    return jsonify({"summary": "摘要生成失败，请检查 API 配置", "key_findings": [], "methodology": "", "limitations": ""})


@app.route('/api/summary/cross', methods=['POST'])
def cross_analyze():
    data = request.get_json() or {}
    paper_ids = data.get("paper_ids", [])
    question = data.get("question", "这些论文的共同挑战是什么？")

    if not paper_ids:
        return jsonify({"analysis": "请选择至少两篇论文"})

    if not _get_llm_config()["api_key"]:
        return jsonify({"analysis": "请先配置 LLM API Key", "comparisons": []})

    paper_summaries = []
    for pid in paper_ids:
        p = _get_store().get(pid)
        if p:
            paper_summaries.append(f"- {p['title']} ({p.get('year','?')}): {p.get('abstract','')[:300]}...")

    papers_text = "\n".join(paper_summaries)
    prompt = f"""Analyze these papers and answer: {question}

Papers:
{papers_text}

Return JSON: {{"analysis": "overall analysis", "comparisons": [{{"dimension": "aspect", "details": {{"paper_title": "value"}}}}]}}"""

    result = _call_llm([
        {"role": "system", "content": "You analyze academic literature. Return only JSON."},
        {"role": "user", "content": prompt},
    ], expect_json=True)

    if result and "error" not in result:
        return jsonify(result)
    return jsonify({"analysis": "分析失败，请检查 API 配置", "comparisons": []})


# ---------------- 研究缺口检测 API ----------------

@app.route('/api/gaps/detect', methods=['POST'])
def detect_gaps():
    papers = _get_store().list_all()
    discussions = []
    for p in papers:
        disc = p.get("sections", {}).get("discussion", "")
        if disc and len(disc) > 100:
            discussions.append({"id": p["id"], "title": p["title"], "discussion": disc[:2000]})

    if len(discussions) < 3:
        return jsonify({"gaps": [], "message": "包含 Discussion 章节的论文不足 3 篇，无法进行缺口检测"})

    if not _get_llm_config()["api_key"]:
        return jsonify({"gaps": [], "message": "请先配置 LLM API Key"})

    disc_text = "\n\n".join([f"Paper: {d['title']}\nDiscussion: {d['discussion']}" for d in discussions])

    prompt = f"""Analyze the Discussion sections of these papers. Identify common limitations and research gaps.
Return JSON: {{"gaps": [{{"gap": "description", "frequency": count, "paper_ids": ["id1","id2"], "suggested_direction": "future research suggestion"}}]}}

Discussion texts:
{disc_text}"""

    result = _call_llm([
        {"role": "system", "content": "You identify research gaps in academic literature. Return only JSON."},
        {"role": "user", "content": prompt},
    ], expect_json=True)

    if result and "gaps" in result:
        return jsonify(result)
    return jsonify({"gaps": [], "message": "缺口检测失败，请检查 API 配置"})


# ---------------- 上下文搭配分析 API（内层新增） ----------------

@app.route('/api/collocation', methods=['POST'])
def collocation():
    """
    分析关键词的前后上下文搭配词。

    输入：{text: "...", keywords: ["词1","词2"], window: 3}
    对每个关键词，在全文找到所有出现位置，取前后各 window 个词，统计频率。
    """
    data = request.get_json() or {}
    text = data.get("text", "")
    keywords = data.get("keywords", [])
    window = data.get("window", 3)

    if not text or not keywords:
        return jsonify({'error': '请提供文本和关键词'}), 400

    raw_words = list(jieba.cut(text))

    result = {}
    for kw in keywords:
        before_counter = collections.Counter()
        after_counter = collections.Counter()

        for i, w in enumerate(raw_words):
            if w == kw:
                for j in range(max(0, i - window), i):
                    context_word = raw_words[j]
                    if len(context_word) > 1 and context_word != kw:
                        before_counter[context_word] += 1
                for j in range(i + 1, min(len(raw_words), i + window + 1)):
                    context_word = raw_words[j]
                    if len(context_word) > 1 and context_word != kw:
                        after_counter[context_word] += 1

        result[kw] = {
            "before": [{"word": w, "count": c} for w, c in before_counter.most_common(15)],
            "after": [{"word": w, "count": c} for w, c in after_counter.most_common(15)],
        }

    return jsonify(result)


# ---------------- 单文本全流程分析 API（保留，向后兼容） ----------------

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # 1. 接收文件与表单数据
        paper_id = request.form.get('paper_id', '')
        if paper_id:
            paper = _get_store().get(paper_id)
            if not paper:
                return jsonify({'error': '论文不存在'}), 404
            content = paper.get("full_text", "")
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '文件名为空'}), 400
            raw_bytes = file.read()
            try:
                content = raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = raw_bytes.decode('gbk', errors='ignore')
        else:
            return jsonify({'error': '未上传文件或未指定论文ID'}), 400

        if not content.strip():
            return jsonify({'error': '文件内容为空'}), 400

        stopwords_raw = request.form.get('stopwords', '')
        custom_stopwords = set([w.strip() for w in re.split(r'[\s,，]+', stopwords_raw) if w.strip()])

        target_word_raw = request.form.get('target_word', '')
        target_words = [w.strip() for w in re.split(r'[\s,，]+', target_word_raw) if w.strip()]

        # 不调用 jieba.add_word，避免多用户环境下的全局副作用
        color_scheme = request.form.get('color_scheme', '科技感')

        # 2. 分词与词性过滤（带降级策略）
        allowed_pos = {'n', 'a', 'nr', 'ns', 'nt', 'nz'}
        words = []
        for item in pseg.lcut(content):
            if len(item.word) > 1 and item.word not in custom_stopwords and item.flag in allowed_pos:
                words.append(item.word)

        # 如果 POS 过滤后词语太少，放宽限制
        if len(words) < 10:
            words = []
            for item in pseg.lcut(content):
                if len(item.word) > 1 and item.word not in custom_stopwords:
                    words.append(item.word)

        processed_text = " ".join(words)

        if not processed_text.strip():
            return jsonify({'error': '有效文本不足（可能全是停用词）'}), 400

        # 3. TF-IDF 分析
        tfidf_results_all = jieba.analyse.extract_tags(processed_text, topK=5000, withWeight=True)
        tfidf_dict = dict(tfidf_results_all)

        tfidf_results = tfidf_results_all[:50]

        global_total_weight = sum(wt for _, wt in tfidf_results_all)
        if global_total_weight == 0:
            global_total_weight = 1.0

        if not tfidf_results:
            return jsonify({'error': '无法提取关键词，文本特征不明显'}), 400

        # 4. 专项词分析
        keyword_stats = []
        total_chars = len(content)
        for kw in target_words:
            count = content.count(kw)
            ratio = (count / max(total_chars, 1) * 100)
            wt = tfidf_dict.get(kw, 0)
            wt_ratio = (wt / global_total_weight * 100)
            weight_str = f"{wt:.4f}" if wt > 0 else "0.0000"

            keyword_stats.append({
                'word': kw, 'count': count, 'ratio': f"{ratio:.2f}%",
                'weight': weight_str, 'weight_val': wt, 'weight_ratio': f"{wt_ratio:.2f}%"
            })

        keyword_stats.sort(key=lambda x: x['weight_val'], reverse=True)

        # 5. 生成词云 (Base64) — 矩形 + 渐变配色
        font_path = get_chinese_font()
        cmap_map = {'科技感': 'Blues', '热力图': 'OrRd', '彩虹色': 'rainbow'}
        cmap_name = cmap_map.get(color_scheme, 'Blues')
        scheme_colors = {'科技感': ['#1e40af', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe'],
                         '热力图': ['#991b1b', '#dc2626', '#ef4444', '#f97316', '#fb923c', '#fdba74'],
                         '彩虹色': ['#7c3aed', '#2563eb', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']}
        colors = scheme_colors.get(color_scheme, scheme_colors['科技感'])

        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            i = hash(word) % len(colors)
            return colors[i]

        wc = WordCloud(
            font_path=font_path, width=1600, height=800,
            background_color=None, mode='RGBA',
            random_state=42, margin=20, max_font_size=150,
            min_font_size=10, max_words=250,
            prefer_horizontal=0.85, repeat=False,
            color_func=color_func,
        ).generate(processed_text)

        img_buffer = io.BytesIO()
        wc.to_image().save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        top_12_words = [item[0] for item in tfidf_results[:12]]
        top_12_weights = [round(item[1], 4) for item in tfidf_results[:12]]
        top_8_pie = [{"name": item[0], "value": round(item[1], 4)} for item in tfidf_results[:8]]

        # 7. 共现网络分析
        sentences = re.split(r'[。！？.!?\n]+', content)
        # 过滤掉常见的无效单字和无意义词
        _noise_words = {'研究', '基于', '本文', '提出', '使用', '进行', '一种', '方法', '分析', '问题',
                        '可以', '不同', '实验', '结果', '表明', '数据', '模型', '通过', '采用', '实现',
                        '需要', '具有', '论文', '提高', '结合', '设计', '其中', '主要', '考虑', '解决',
                        '利用', '以及', '这个', '他们', '我们', '非常', '比较', '一个', '一些', '之间',
                        '能够', '不能', '存在', '提供', '验证', '计算', '过程', '影响', '改变', '成为'}
        top_30_words = [item[0] for item in tfidf_results[:50] if len(item[0]) >= 2 and item[0] not in _noise_words][:30]
        co_matrix = collections.defaultdict(int)

        processed_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            sen_words = [w for w in jieba.lcut(sentence) if w in top_30_words]
            processed_sentences.append(" ".join(sen_words))
            sen_words = list(set(sen_words))
            for i in range(len(sen_words)):
                for j in range(i + 1, len(sen_words)):
                    w1, w2 = sorted([sen_words[i], sen_words[j]])
                    co_matrix[(w1, w2)] += 1

        tfidf_max = max(wt for _, wt in tfidf_results_all[:50]) or 0.01
        graph_nodes = [{"name": w, "symbolSize": max(18, (dict(tfidf_results).get(w, 0) / tfidf_max) * 55)} for w in top_30_words]

        all_links = []
        for (w1, w2), count in co_matrix.items():
            if count >= 2:
                all_links.append({"source": w1, "target": w2, "value": count})
        all_links.sort(key=lambda x: x['value'], reverse=True)
        graph_links = all_links[:30]

        # 8. 语义聚类
        tree_data = {"name": "全局语义", "children": []}
        top_25_words = [item[0] for item in tfidf_results[:25]]
        if len(processed_sentences) > 5 and len(top_25_words) >= 5:
            vectorizer = TfidfVectorizer(vocabulary=top_25_words)
            tfidf_matrix = vectorizer.fit_transform(processed_sentences).T

            num_clusters = min(4, len(top_25_words) // 4)
            num_clusters = max(2, num_clusters)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix.toarray())

            clusters = collections.defaultdict(list)
            for word, label in zip(top_25_words, kmeans.labels_):
                clusters[label].append((word, dict(tfidf_results).get(word, 0)))

            for label, words_in_cluster in clusters.items():
                words_in_cluster.sort(key=lambda x: x[1], reverse=True)
                cluster_core = words_in_cluster[0][0]
                cluster_node = {
                    "name": f"核心: {cluster_core}",
                    "children": [{"name": w, "value": round(wt, 4)} for w, wt in words_in_cluster]
                }
                tree_data["children"].append(cluster_node)

        # 9. 构建监测词专属数据
        valid_targets = [kw for kw in target_words if kw in content]

        monitor_bar_data = {
            'categories': [stat['word'] for stat in keyword_stats],
            'values': [stat['weight_val'] for stat in keyword_stats]
        }

        monitor_pie_data = [{"name": stat['word'], "value": stat['weight_val']} for stat in keyword_stats if stat['weight_val'] > 0]

        m_co_matrix = collections.defaultdict(int)
        for sentence in sentences:
            if not sentence.strip():
                continue
            sen_words = [w for w in jieba.lcut(sentence) if w in words]
            for tw in valid_targets:
                if tw in sentence and tw not in sen_words:
                    sen_words.append(tw)
            has_target = any(w in valid_targets for w in sen_words)
            if not has_target:
                continue
            context_words = set(valid_targets + top_30_words)
            sen_words = list(set([w for w in sen_words if w in context_words]))
            for i in range(len(sen_words)):
                for j in range(i + 1, len(sen_words)):
                    w1, w2 = sorted([sen_words[i], sen_words[j]])
                    if w1 in valid_targets or w2 in valid_targets:
                        m_co_matrix[(w1, w2)] += 1

        m_links = []
        for (w1, w2), count in m_co_matrix.items():
            if count >= 1:
                m_links.append({"source": w1, "target": w2, "value": count})
        m_links.sort(key=lambda x: x['value'], reverse=True)
        m_links = m_links[:40]

        m_nodes_set = set(valid_targets)
        for link in m_links:
            m_nodes_set.add(link['source'])
            m_nodes_set.add(link['target'])

        monitor_graph_nodes = [{"name": w,
                                 "symbolSize": max(10, tfidf_dict.get(w, 0) * 80 if w not in valid_targets else tfidf_dict.get(w, 0) * 120 + 20)}
                                for w in m_nodes_set]

        monitor_tree_data = {"name": "监测词语义聚类", "children": []}
        m_tree_words = list(m_nodes_set)
        if len(m_tree_words) >= 2 and len(processed_sentences) > 2:
            try:
                m_vectorizer = TfidfVectorizer(vocabulary=m_tree_words)
                m_tfidf_matrix = m_vectorizer.fit_transform(processed_sentences).T
                m_num_clusters = min(3, len(m_tree_words) // 2)
                m_num_clusters = max(2, m_num_clusters)

                m_kmeans = KMeans(n_clusters=m_num_clusters, random_state=42, n_init=10)
                m_kmeans.fit(m_tfidf_matrix.toarray())

                m_clusters = collections.defaultdict(list)
                for word, label in zip(m_tree_words, m_kmeans.labels_):
                    m_clusters[label].append((word, tfidf_dict.get(word, 0)))

                for label, words_in_cluster in m_clusters.items():
                    words_in_cluster.sort(key=lambda x: x[1], reverse=True)
                    cluster_core = words_in_cluster[0][0]
                    cluster_node = {
                        "name": f"关联: {cluster_core}",
                        "children": [{"name": w, "value": round(wt, 4)} for w, wt in words_in_cluster]
                    }
                    monitor_tree_data["children"].append(cluster_node)
            except Exception:
                monitor_tree_data = {"name": "数据不足无法聚类", "children": []}
        else:
            monitor_tree_data = {"name": "监测词共现数据不足", "children": []}

        return jsonify({
            'success': True,
            'wordcloud': f"data:image/png;base64,{img_str}",
            'bar_data': {'categories': top_12_words, 'values': top_12_weights},
            'pie_data': top_8_pie,
            'graph_data': {'nodes': graph_nodes, 'links': graph_links},
            'tree_data': tree_data,
            'monitor_bar_data': {'categories': monitor_bar_data['categories'], 'values': monitor_bar_data['values']},
            'monitor_pie_data': monitor_pie_data,
            'monitor_graph_data': {'nodes': monitor_graph_nodes, 'links': m_links},
            'monitor_tree_data': monitor_tree_data,
            'keyword_stats': keyword_stats,
            'all_words': [{"word": item[0], "weight": round(item[1], 4)} for item in tfidf_results]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------- 启动 ----------------

if __name__ == '__main__':
    app.run(debug=True, port=5000)
