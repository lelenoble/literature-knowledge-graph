import time
from models.paper import Paper
from models.relation import Relation


class PaperStore:
    def __init__(self):
        self._papers: dict[str, dict] = {}
        self._relations: list[Relation] = []
        self._next_id: int = 1

    def add(self, paper_data: dict) -> str:
        pid = str(self._next_id)
        self._next_id += 1

        paper = {
            "id": pid,
            "filename": paper_data.get("filename", ""),
            "title": paper_data.get("title", "未命名"),
            "authors": paper_data.get("authors", []),
            "year": paper_data.get("year"),
            "abstract": paper_data.get("abstract", ""),
            "full_text": paper_data.get("full_text", ""),
            "keywords": paper_data.get("keywords", []),
            "sections": paper_data.get("sections", {}),
            "language": paper_data.get("language", "zh"),
            "upload_time": time.time(),
            "cluster_id": paper_data.get("cluster_id"),
        }
        self._papers[pid] = paper
        return pid

    def get(self, paper_id: str) -> dict | None:
        return self._papers.get(paper_id)

    def list_all(self) -> list[dict]:
        return sorted(
            self._papers.values(),
            key=lambda p: p.get("year") or 0,
            reverse=True,
        )

    def remove(self, paper_id: str) -> bool:
        if paper_id in self._papers:
            del self._papers[paper_id]
            self._relations = [
                r for r in self._relations
                if r.source_id != paper_id and r.target_id != paper_id
            ]
            return True
        return False

    def clear(self):
        self._papers.clear()
        self._relations.clear()

    def add_relation(self, relation: Relation):
        existing = [
            r for r in self._relations
            if r.source_id == relation.source_id and r.target_id == relation.target_id
        ]
        if existing:
            existing[0] = relation
        else:
            self._relations.append(relation)

    def get_relations(self) -> list[Relation]:
        return list(self._relations)

    def get_relations_for_paper(self, paper_id: str) -> list[Relation]:
        return [
            r for r in self._relations
            if r.source_id == paper_id or r.target_id == paper_id
        ]

    def clear_relations(self):
        self._relations.clear()

    @property
    def paper_count(self) -> int:
        return len(self._papers)

    @property
    def relation_count(self) -> int:
        return len(self._relations)
