from dataclasses import dataclass, field
from enum import Enum


class RelationType(Enum):
    CITES = "cites"
    IMPROVES = "improves"
    REFUTES = "refutes"
    SHARED_METHOD = "shared_method"
    SHARED_TOPIC = "shared_topic"


RELATION_LABELS = {
    RelationType.CITES: "引用",
    RelationType.IMPROVES: "改进",
    RelationType.REFUTES: "反驳",
    RelationType.SHARED_METHOD: "共享方法",
    RelationType.SHARED_TOPIC: "同主题",
}

RELATION_COLORS = {
    RelationType.CITES: "#3b82f6",
    RelationType.IMPROVES: "#22c55e",
    RelationType.REFUTES: "#ef4444",
    RelationType.SHARED_METHOD: "#f59e0b",
    RelationType.SHARED_TOPIC: "#8b5cf6",
}


@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float = 1.0
    evidence: str = ""
