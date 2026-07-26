"""
AstroNexus AI — Neo4j Graph Seeder

Pre-loads domain nodes, keyword tag nodes, and their relationships
into Neo4j before any papers are ingested.

This creates the vocabulary scaffold that paper ingestion fills in.

Run once:
    python -m backend.graph.seed_graph

Safe to re-run — uses MERGE so nothing is duplicated.
"""
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── Domain definitions with display names and keywords ────────────────────────

DOMAINS = {
    "remote_sensing": {
        "display_name": "Remote Sensing",
        "description":  "Satellite imagery, earth observation, geospatial analysis",
        "keywords": [
            "satellite imagery", "land use classification", "land cover mapping",
            "change detection", "flood detection", "vegetation index", "ndvi",
            "spectral analysis", "multispectral", "hyperspectral", "lidar",
            "synthetic aperture radar", "sar", "geotiff", "object detection",
            "semantic segmentation", "scene classification", "crop monitoring",
            "urban mapping", "coastal monitoring", "deforestation detection",
            "disaster mapping", "earth observation", "remote sensing",
        ],
    },
    "nlp": {
        "display_name": "Natural Language Processing",
        "description":  "Language models, text understanding, generation",
        "keywords": [
            "language model", "text classification", "machine translation",
            "question answering", "summarization", "named entity recognition",
            "sentiment analysis", "text generation", "sequence labeling",
            "information extraction", "dialogue systems", "reading comprehension",
        ],
    },
    "computer_vision": {
        "display_name": "Computer Vision",
        "description":  "Image understanding, object detection, segmentation",
        "keywords": [
            "image classification", "object detection", "image segmentation",
            "instance segmentation", "panoptic segmentation", "depth estimation",
            "optical flow", "pose estimation", "face recognition",
            "image generation", "visual grounding", "image captioning",
        ],
    },
    "machine_learning": {
        "display_name": "Machine Learning",
        "description":  "Deep learning, neural networks, training methods",
        "keywords": [
            "deep learning", "transfer learning", "self-supervised learning",
            "contrastive learning", "federated learning", "meta-learning",
            "knowledge distillation", "neural architecture search",
            "reinforcement learning", "generative models", "diffusion models",
        ],
    },
    "astronomy": {
        "display_name": "Astronomy",
        "description":  "Space observation, astrophysics, cosmology",
        "keywords": [
            "galaxy classification", "star detection", "exoplanet detection",
            "gravitational wave detection", "spectral analysis",
            "photometric redshift", "cosmic microwave background",
            "telescope image analysis",
        ],
    },
    "medicine": {
        "display_name": "Medicine",
        "description":  "Medical imaging, clinical AI, bioinformatics",
        "keywords": [
            "medical image segmentation", "disease classification",
            "drug discovery", "clinical prediction", "radiology",
            "pathology detection", "genomic analysis", "protein structure",
        ],
    },
    "climate": {
        "display_name": "Climate Science",
        "description":  "Climate modelling, weather prediction, environmental monitoring",
        "keywords": [
            "climate modelling", "weather forecasting", "carbon monitoring",
            "sea level prediction", "extreme weather detection",
            "atmospheric modelling", "ocean temperature analysis",
        ],
    },
}


def seed(clear_existing: bool = False) -> None:
    """
    Write all domain and keyword nodes to Neo4j.

    Args:
        clear_existing: If True, delete existing Domain and Tag nodes first.
                        Use only during development resets.
    """
    from backend.graph.neo4j_client import _get_driver, create_constraints

    driver = _get_driver()

    # Ensure uniqueness constraints exist
    create_constraints()

    with driver.session() as session:

        if clear_existing:
            session.run("MATCH (d:Domain) DETACH DELETE d")
            session.run("MATCH (t:Tag) DETACH DELETE t")
            logger.info("Cleared existing Domain and Tag nodes.")

        total_domains  = 0
        total_keywords = 0

        for domain_key, domain_data in DOMAINS.items():
            # ── Create Domain node ─────────────────────────────────────────────
            session.run(
                """
                MERGE (d:Domain {key: $key})
                SET d.name        = $name,
                    d.description = $description
                """,
                key=         domain_key,
                name=        domain_data["display_name"],
                description= domain_data["description"],
            )
            total_domains += 1

            # ── Create Tag nodes and link to Domain ────────────────────────────
            for kw in domain_data["keywords"]:
                session.run(
                    """
                    MERGE (t:Tag {name: $kw})
                    WITH t
                    MATCH (d:Domain {key: $domain_key})
                    MERGE (d)-[:HAS_TAG]->(t)
                    """,
                    kw=         kw,
                    domain_key= domain_key,
                )
                total_keywords += 1

            logger.info(
                f"  ✓ {domain_data['display_name']:<30} "
                f"{len(domain_data['keywords'])} keywords"
            )

    logger.info(
        f"\nSeeded {total_domains} domains and "
        f"{total_keywords} keyword tags into Neo4j."
    )


def verify() -> None:
    """Print a summary of what was seeded."""
    from backend.graph.neo4j_client import _get_driver

    driver = _get_driver()
    with driver.session() as session:
        domains = session.run(
            "MATCH (d:Domain) RETURN d.name AS name, d.key AS key ORDER BY name"
        ).data()
        tags = session.run("MATCH (t:Tag) RETURN count(t) AS n").single()

    print(f"\nNeo4j graph seed verification:")
    print(f"  Domains : {len(domains)}")
    for d in domains:
        print(f"    {d['name']}")
    print(f"  Tags    : {tags['n']}")


def run() -> None:
    print("\n" + "="*50)
    print("ASTRONEXUS — NEO4J GRAPH SEEDER")
    print("="*50)

    try:
        seed()
        verify()
        print("\n✓ Graph seeded successfully.")
        print("  Papers ingested via /upload will now be tagged to these domains.")
    except Exception as e:
        print(f"\n✗ Seeding failed: {e}")
        print("  Make sure Neo4j is running: docker start astronexus-neo4j")
        sys.exit(1)


if __name__ == "__main__":
    run()