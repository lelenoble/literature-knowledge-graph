from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    id: str
    filename: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    full_text: str = ""
    keywords: list[str] = field(default_factory=list)
    sections: dict = field(default_factory=dict)
    language: str = "zh"
    upload_time: float = 0.0
    cluster_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract[:500],
            "full_text": self.full_text,
            "keywords": self.keywords,
            "sections": {k: v[:300] for k, v in self.sections.items()},
            "language": self.language,
            "upload_time": self.upload_time,
            "cluster_id": self.cluster_id,
        }
