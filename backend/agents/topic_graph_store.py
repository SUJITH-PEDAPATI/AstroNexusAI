import hashlib
from datetime import datetime
from dataclasses import dataclass
from backend.graph.neo4j_client import _get_driver
from backend.agents.tier_classifier import extract_keywords

@dataclass
class TopicResult:
    topic_id: str
    topic_label: str
    is_new: bool
    keywords: list[str]

def generate_topic_id(keywords: list[str]) -> str:
    if not keywords:
        return "topic_general"
    
    # Use the primary keyword (often the most significant phrase prepended by extract_keywords)
    # This ensures queries sharing the main entity (like 'black hole') group together.
    primary_kw = keywords[0].lower()
    if primary_kw == 'black':  # Fallback just in case
        primary_kw = 'black hole'
        
    hash_str = primary_kw
    return "topic_" + hashlib.md5(hash_str.encode()).hexdigest()[:8]

def store_query_topic(query: str) -> TopicResult:
    keywords = extract_keywords(query)
    topic_id = generate_topic_id(keywords)
    topic_label = keywords[0].title() if keywords else "General"
    
    driver = _get_driver()
    with driver.session() as s:
        # Check if topic already exists
        record = s.run("MATCH (t:Topic {topic_id: $tid}) RETURN t", tid=topic_id).single()
        is_new = record is None
        
        if is_new:
            s.run("""
                MERGE (t:Topic {topic_id: $tid})
                SET t.label = $label,
                    t.created_at = $ts
            """, tid=topic_id, label=topic_label, ts=datetime.utcnow().isoformat())
            
        # Create topic subgraph mapping to keywords
        for kw in keywords[:5]:
            s.run("""
                MATCH (t:Topic {topic_id: $tid})
                MERGE (k:Keyword {name: $kw})
                MERGE (t)-[:HAS_KEYWORD]->(k)
            """, tid=topic_id, kw=kw)
                
    return TopicResult(
        topic_id=topic_id,
        topic_label=topic_label,
        is_new=is_new,
        keywords=keywords
    )

def find_topic_by_query(query: str) -> str | None:
    keywords = extract_keywords(query)
    return generate_topic_id(keywords)

def get_topic_graph(topic_id: str) -> list[dict]:
    driver = _get_driver()
    with driver.session() as s:
        rows = s.run("""
            MATCH (t:Topic {topic_id: $tid})-[:HAS_KEYWORD]->(k:Keyword)
            RETURN k.name AS target_name
        """, tid=topic_id).data()
    return rows

def list_topics() -> list[dict]:
    driver = _get_driver()
    with driver.session() as s:
        rows = s.run("""
            MATCH (t:Topic)
            RETURN t.topic_id AS topic_id, t.label AS label
        """).data()
    return rows
