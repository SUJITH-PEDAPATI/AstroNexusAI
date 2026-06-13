from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field
from uuid import uuid4


# ── Node Types ─────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    PAPER   = "Paper"
    AUTHOR  = "Author"
    MODEL   = "Model"
    DATASET = "Dataset"
    TASK    = "Task"
    VENUE   = "Venue"       # conference / journal


class RelationType(str, Enum):
    AUTHORED_BY = "AUTHORED_BY"    # Paper → Author
    USES        = "USES"           # Paper → Model | Dataset
    IMPROVES    = "IMPROVES"       # Paper → Paper
    COMPARES    = "COMPARES"       # Paper → Model
    DEPENDS_ON  = "DEPENDS_ON"     # Model → Model
    PUBLISHED_AT = "PUBLISHED_AT"  # Paper → Venue
    SOLVES      = "SOLVES"         # Paper → Task


# ── Node Models ────────────────────────────────────────────────────────────────

class PaperNode(BaseModel):
    node_id:    str = Field(default_factory=lambda: str(uuid4()))
    title:      str
    year:       int | None = None
    doi:        str | None = None
    arxiv_id:   str | None = None
    abstract:   str | None = None
    source_file: str | None = None
    paper_id:   str | None = None   # links back to RawDocument.paper_id


class AuthorNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    name:    str


class ModelNode(BaseModel):
    node_id:     str = Field(default_factory=lambda: str(uuid4()))
    name:        str
    description: str | None = None


class DatasetNode(BaseModel):
    node_id:     str = Field(default_factory=lambda: str(uuid4()))
    name:        str
    description: str | None = None


class TaskNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    name:    str


class VenueNode(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    name:    str
    year:    int | None = None


# ── Relation Model ─────────────────────────────────────────────────────────────

class GraphRelation(BaseModel):
    from_id:       str                # node_id of source node
    to_id:         str                # node_id of target node
    relation_type: RelationType
    properties:    dict = Field(default_factory=dict)


# ── Extraction Result ──────────────────────────────────────────────────────────

class ExtractionResult(BaseModel):
    """
    All entities and relations extracted from a single paper.
    Output of entity_extractor.py, input to graph_builder.py.
    """
    paper:    PaperNode
    authors:  list[AuthorNode]   = Field(default_factory=list)
    models:   list[ModelNode]    = Field(default_factory=list)
    datasets: list[DatasetNode]  = Field(default_factory=list)
    tasks:    list[TaskNode]     = Field(default_factory=list)
    venues:   list[VenueNode]    = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)