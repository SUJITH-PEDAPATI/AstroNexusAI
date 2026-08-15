"""
AstroNexus AI — Voice Entity Graph Store
==========================================
Writes voice-transcript entities into the existing Neo4j knowledge graph.

Separate from query_keyword_store.py (which writes query-level keywords).
This module writes richer entity data extracted from full voice transcripts.

Graph additions (all MERGE — never duplicates):

    (:VoiceSession {session_id, timestamp, duration_sec, language})
        │
        ├─[:MENTIONED]──► (:Entity {name, label, source:"voice"})
        │                     │
        │                     └─[:IS_A]──► (:EntityType {name:"ORG"/"MISSION"/etc})
        │
        └─[:TRANSCRIBED]─► (:VoiceTranscript {text, word_count, timestamp})

Preserves:
    • Existing Paper / Author / Model / Dataset / Task / Venue schema
    • Existing Keyword / QuerySession / Domain nodes
    • All existing constraints and indices

Uses MERGE throughout — safe to call multiple times with the same data.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.voice.keyword_extractor import Entity, KeywordResult

logger = logging.getLogger(__name__)


# ── MERGE-safe Cypher helpers ─────────────────────────────────────────────────

def _run(cypher: str, **params) -> None:
    """Execute a Cypher statement — non-fatal; logs on failure."""
    try:
        from backend.graph.neo4j_client import _get_driver
        driver = _get_driver()
        with driver.session() as s:
            s.run(cypher, **params)
    except Exception as e:
        logger.warning(f"[VoiceGraphStore] Cypher failed: {e}\nQuery: {cypher[:80]}")


# ── Entity label → Neo4j node label mapping ───────────────────────────────────

_LABEL_MAP: dict[str, str] = {
    "ORG":         "Organization",
    "PERSON":      "Person",
    "GPE":         "Location",
    "LOC":         "Location",
    "PRODUCT":     "Product",
    "EVENT":       "Event",
    "WORK_OF_ART": "Mission",
    "NORP":        "Group",
    "PATTERN":     "ScientificTerm",
}


# ── Public API ────────────────────────────────────────────────────────────────

def store_voice_session(
    session_id:   str,
    transcript:   str,
    keywords:     "KeywordResult",
    duration_sec: float = 0.0,
    language:     str   = "en",
    paper_ids:    list[str] | None = None,
) -> dict:
    """
    Write the complete voice session and its extracted entities to Neo4j.

    Args:
        session_id:   Unique identifier for this recording session
        transcript:   Full, clean transcript text
        keywords:     KeywordResult from keyword_extractor.extract()
        duration_sec: Recording duration
        language:     Detected language code

    Returns:
        Summary dict: {session_id, entities_written, keywords_written, status}
    """
    ts = datetime.now(timezone.utc).isoformat()

    # ── 1. VoiceSession node ──────────────────────────────────────────────────
    _run(
        """
        MERGE (vs:VoiceSession {session_id: $sid})
        SET vs.timestamp    = $ts,
            vs.duration_sec = $dur,
            vs.language     = $lang,
            vs.word_count   = $wc
        """,
        sid=  session_id,
        ts=   ts,
        dur=  round(duration_sec, 2),
        lang= language,
        wc=   len(transcript.split()),
    )
    logger.info(f"[VoiceGraphStore] Session node: {session_id}")

    # ── 2. VoiceTranscript node ───────────────────────────────────────────────
    _run(
        """
        MERGE (vt:VoiceTranscript {session_id: $sid})
        SET vt.text      = $text,
            vt.timestamp = $ts
        WITH vt
        MATCH (vs:VoiceSession {session_id: $sid})
        MERGE (vs)-[:TRANSCRIBED]->(vt)
        """,
        sid=  session_id,
        text= transcript[:2000],   # cap at 2KB per node
        ts=   ts,
    )

    # ── 2b. Link VoiceSession → Paper (RELATED_TO) ──────────────────────────
    for pid in (paper_ids or []):
        _run(
            """
            MERGE (vi:VoiceInput {voice_id: $sid})
            SET vi.text      = $text,
                vi.timestamp = $ts
            WITH vi
            MATCH (p:Paper {paper_id: $pid})
            MERGE (vi)-[:RELATED_TO]->(p)
            """,
            sid=session_id,
            text=transcript[:500],
            ts=datetime.now(timezone.utc).isoformat(),
            pid=pid,
        )

    # ── 3. Entity nodes ───────────────────────────────────────────────────────
    entities_written = 0
    for entity in keywords.entities:
        node_label = _LABEL_MAP.get(entity.label, "Entity")
        name_key   = entity.name.strip()
        if not name_key:
            continue

        # MERGE entity node
        _run(
            f"""
            MERGE (e:{node_label} {{name: $name}})
            SET e.source       = 'voice',
                e.last_seen    = $ts,
                e.mention_count = coalesce(e.mention_count, 0) + 1
            """,
            name= name_key,
            ts=   ts,
        )

        # MERGE EntityType (for graph browsing)
        _run(
            """
            MERGE (et:EntityType {name: $label})
            WITH et
            MATCH (e {name: $name})
            MERGE (e)-[:IS_A]->(et)
            """,
            label= entity.label,
            name=  name_key,
        )

        # Link session → entity (MENTIONED)
        _run(
            f"""
            MATCH (vs:VoiceSession {{session_id: $sid}})
            MATCH (e:{node_label} {{name: $name}})
            MERGE (vs)-[:MENTIONED]->(e)
            """,
            sid=  session_id,
            name= name_key,
        )
        # Also write as VoiceInput MENTIONS Entity (required schema)
        _run(
            f"""
            MERGE (vi:VoiceInput {{voice_id: $sid}})
            MERGE (e:Entity {{name: $name}})
            SET e.type = $etype
            WITH vi, e
            MERGE (vi)-[:MENTIONS]->(e)
            """,
            sid=  session_id,
            name= name_key,
            etype=node_label,
        )
        entities_written += 1

    # ── 4. Scientific term nodes ──────────────────────────────────────────────
    keywords_written = 0
    for term in keywords.scientific:
        if not term.strip():
            continue
        _run(
            """
            MERGE (st:ScientificTerm {name: $name})
            SET st.source       = 'voice',
                st.last_seen    = $ts,
                st.mention_count = coalesce(st.mention_count, 0) + 1
            WITH st
            MATCH (vs:VoiceSession {session_id: $sid})
            MERGE (vs)-[:MENTIONED]->(st)
            """,
            name= term.strip().title(),
            ts=   ts,
            sid=  session_id,
        )
        keywords_written += 1

    # ── 5. Plain keywords (reuse existing Keyword schema) ────────────────────
    for kw in keywords.keywords[:20]:    # cap to avoid flooding graph
        if not kw.strip() or len(kw) < 3:
            continue
        _run(
            """
            MERGE (k:Keyword {name: $kw})
            SET k.source       = 'voice',
                k.last_queried = $ts,
                k.query_count  = coalesce(k.query_count, 0) + 1
            WITH k
            MATCH (vs:VoiceSession {session_id: $sid})
            MERGE (vs)-[:ASKED_ABOUT]->(k)
            """,
            kw=  kw.lower().strip(),
            ts=  ts,
            sid= session_id,
        )
        keywords_written += 1

    summary = {
        "session_id":      session_id,
        "entities_written": entities_written,
        "keywords_written": keywords_written,
        "status":          "ok",
    }
    logger.info(
        f"[VoiceGraphStore] entities={entities_written} "
        f"keywords={keywords_written} session={session_id}"
    )
    return summary