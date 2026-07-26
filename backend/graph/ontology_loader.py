"""
AstroNexus AI — Remote Sensing Ontology Loader

Initialises Neo4j with a prebuilt domain knowledge scaffold for
remote sensing, earth observation, and geospatial AI.

Design principles:
    • No hardcoded thousands of entities — structured dicts only
    • Modular: add new domains by adding entries to ONTOLOGY
    • Idempotent: MERGE ensures safe re-runs
    • Future-proof: load external JSON/YAML ontology files later

Node types created:
    Institution, Satellite, Sensor, Mission,
    Algorithm, Metric, LossFunction, Conference, Journal, Keyword

Relationships created:
    HAS_SENSOR       Satellite → Sensor
    PART_OF_MISSION  Satellite → Mission
    OPERATED_BY      Mission   → Institution
    RELATED_TO       Keyword   → Domain

Run once:
    python -m backend.graph.ontology_loader

Safe to re-run — MERGE prevents duplicates.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# ONTOLOGY DEFINITION
# Each section is independently loadable.
# Add new sections without touching existing ones.
# ══════════════════════════════════════════════════════════════════════════════

INSTITUTIONS: list[dict] = [
    {"name": "NASA",  "full_name": "National Aeronautics and Space Administration", "country": "USA"},
    {"name": "ESA",   "full_name": "European Space Agency",                        "country": "Europe"},
    {"name": "ISRO",  "full_name": "Indian Space Research Organisation",           "country": "India"},
    {"name": "JAXA",  "full_name": "Japan Aerospace Exploration Agency",           "country": "Japan"},
    {"name": "CNES",  "full_name": "Centre National d'Études Spatiales",           "country": "France"},
    {"name": "DLR",   "full_name": "German Aerospace Center",                      "country": "Germany"},
    {"name": "USGS",  "full_name": "United States Geological Survey",              "country": "USA"},
    {"name": "NOAA",  "full_name": "National Oceanic and Atmospheric Administration","country": "USA"},
]

MISSIONS: list[dict] = [
    {"name": "Copernicus",        "operator": "ESA",  "purpose": "Earth observation"},
    {"name": "Landsat Program",   "operator": "NASA", "purpose": "Land surface monitoring"},
    {"name": "MODIS",             "operator": "NASA", "purpose": "Global surface monitoring"},
    {"name": "Sentinel Program",  "operator": "ESA",  "purpose": "Multi-purpose earth observation"},
    {"name": "Planet Labs",       "operator": None,   "purpose": "Commercial high-resolution imaging"},
    {"name": "NISAR",             "operator": "NASA", "purpose": "SAR earth observation"},
    {"name": "Chandrayaan",       "operator": "ISRO", "purpose": "Lunar exploration"},
    {"name": "INSAT",             "operator": "ISRO", "purpose": "Meteorology and communication"},
]

SATELLITES: list[dict] = [
    {"name": "Sentinel-1",  "mission": "Sentinel Program", "type": "SAR",          "resolution_m": 5},
    {"name": "Sentinel-2",  "mission": "Sentinel Program", "type": "Multispectral","resolution_m": 10},
    {"name": "Sentinel-3",  "mission": "Sentinel Program", "type": "Ocean/Land",   "resolution_m": 300},
    {"name": "Landsat-8",   "mission": "Landsat Program",  "type": "Multispectral","resolution_m": 30},
    {"name": "Landsat-9",   "mission": "Landsat Program",  "type": "Multispectral","resolution_m": 30},
    {"name": "MODIS Terra", "mission": "MODIS",            "type": "Multispectral","resolution_m": 250},
    {"name": "MODIS Aqua",  "mission": "MODIS",            "type": "Multispectral","resolution_m": 250},
    {"name": "WorldView-3", "mission": None,               "type": "Commercial",   "resolution_m": 0.31},
]

SENSORS: list[dict] = [
    {"name": "SAR",           "full_name": "Synthetic Aperture Radar",    "wavelength": "microwave"},
    {"name": "MSI",           "full_name": "Multispectral Instrument",    "wavelength": "visible-NIR"},
    {"name": "OLI",           "full_name": "Operational Land Imager",     "wavelength": "visible-SWIR"},
    {"name": "TIRS",          "full_name": "Thermal Infrared Sensor",     "wavelength": "thermal"},
    {"name": "MODIS Sensor",  "full_name": "Moderate Resolution Imaging Spectroradiometer", "wavelength": "visible-TIR"},
    {"name": "LiDAR",         "full_name": "Light Detection and Ranging", "wavelength": "laser"},
    {"name": "Hyperspectral", "full_name": "Hyperspectral Imager",        "wavelength": "400-2500nm"},
]

SATELLITE_SENSOR_MAP: list[dict] = [
    {"satellite": "Sentinel-1",  "sensor": "SAR"},
    {"satellite": "Sentinel-2",  "sensor": "MSI"},
    {"satellite": "Landsat-8",   "sensor": "OLI"},
    {"satellite": "Landsat-8",   "sensor": "TIRS"},
    {"satellite": "Landsat-9",   "sensor": "OLI"},
    {"satellite": "Landsat-9",   "sensor": "TIRS"},
    {"satellite": "MODIS Terra", "sensor": "MODIS Sensor"},
    {"satellite": "MODIS Aqua",  "sensor": "MODIS Sensor"},
]

ALGORITHMS: list[dict] = [
    # Vision foundation models
    {"name": "DINOv2",       "type": "Feature Extractor",   "domain": "remote_sensing"},
    {"name": "SAM2",         "type": "Segmentation",        "domain": "remote_sensing"},
    {"name": "CLIP",         "type": "Vision-Language",     "domain": "remote_sensing"},
    {"name": "Florence-2",   "type": "Vision-Language",     "domain": "remote_sensing"},
    {"name": "U-Net",        "type": "Segmentation",        "domain": "remote_sensing"},
    {"name": "SegFormer",    "type": "Segmentation",        "domain": "remote_sensing"},
    {"name": "YOLO",         "type": "Object Detection",    "domain": "remote_sensing"},
    {"name": "Faster R-CNN", "type": "Object Detection",    "domain": "remote_sensing"},
    {"name": "DeepLab",      "type": "Segmentation",        "domain": "remote_sensing"},
    {"name": "SpectralGPT",  "type": "Spectral Foundation", "domain": "remote_sensing"},
    # NLP / LLM
    {"name": "Transformer",  "type": "Architecture",        "domain": "nlp"},
    {"name": "BERT",         "type": "Language Model",      "domain": "nlp"},
    {"name": "GPT",          "type": "Language Model",      "domain": "nlp"},
    {"name": "T5",           "type": "Seq2Seq Model",       "domain": "nlp"},
    {"name": "BGE-M3",       "type": "Embedding Model",     "domain": "nlp"},
    # General ML
    {"name": "Vision Transformer", "type": "Architecture", "domain": "computer_vision"},
    {"name": "ResNet",       "type": "CNN Architecture",    "domain": "computer_vision"},
    {"name": "EfficientNet", "type": "CNN Architecture",    "domain": "computer_vision"},
]

METRICS: list[dict] = [
    {"name": "mIoU",       "full_name": "Mean Intersection over Union", "task": "segmentation"},
    {"name": "OA",         "full_name": "Overall Accuracy",             "task": "classification"},
    {"name": "AA",         "full_name": "Average Accuracy",             "task": "classification"},
    {"name": "F1 Score",   "full_name": "F1 Score",                     "task": "classification"},
    {"name": "BLEU",       "full_name": "Bilingual Evaluation Understudy","task": "generation"},
    {"name": "ROUGE-L",    "full_name": "Recall-Oriented Understudy for Gisting Evaluation", "task": "generation"},
    {"name": "MAP",        "full_name": "Mean Average Precision",       "task": "detection"},
    {"name": "NDCG",       "full_name": "Normalized Discounted Cumulative Gain", "task": "retrieval"},
    {"name": "PSNR",       "full_name": "Peak Signal-to-Noise Ratio",   "task": "super_resolution"},
    {"name": "SSIM",       "full_name": "Structural Similarity Index",  "task": "super_resolution"},
    {"name": "Kappa",      "full_name": "Cohen's Kappa Coefficient",    "task": "classification"},
    {"name": "AUC-ROC",    "full_name": "Area Under ROC Curve",         "task": "classification"},
]

LOSS_FUNCTIONS: list[dict] = [
    {"name": "Cross-Entropy Loss",     "use_case": "classification"},
    {"name": "Focal Loss",             "use_case": "imbalanced classification"},
    {"name": "Dice Loss",              "use_case": "segmentation"},
    {"name": "IoU Loss",               "use_case": "segmentation"},
    {"name": "MSE Loss",               "use_case": "regression"},
    {"name": "Contrastive Loss",       "use_case": "self-supervised learning"},
    {"name": "Triplet Loss",           "use_case": "metric learning"},
    {"name": "Binary Cross-Entropy",   "use_case": "binary classification"},
]

CONFERENCES: list[dict] = [
    {"name": "IGARSS",    "full_name": "International Geoscience and Remote Sensing Symposium",  "domain": "remote_sensing"},
    {"name": "CVPR",      "full_name": "Conference on Computer Vision and Pattern Recognition",   "domain": "computer_vision"},
    {"name": "NeurIPS",   "full_name": "Neural Information Processing Systems",                   "domain": "machine_learning"},
    {"name": "ICLR",      "full_name": "International Conference on Learning Representations",    "domain": "machine_learning"},
    {"name": "ICML",      "full_name": "International Conference on Machine Learning",            "domain": "machine_learning"},
    {"name": "ECCV",      "full_name": "European Conference on Computer Vision",                  "domain": "computer_vision"},
    {"name": "ICCV",      "full_name": "International Conference on Computer Vision",             "domain": "computer_vision"},
    {"name": "ACL",       "full_name": "Association for Computational Linguistics",               "domain": "nlp"},
    {"name": "EMNLP",     "full_name": "Empirical Methods in Natural Language Processing",        "domain": "nlp"},
    {"name": "ISPRS",     "full_name": "International Society for Photogrammetry and Remote Sensing","domain": "remote_sensing"},
]

JOURNALS: list[dict] = [
    {"name": "Remote Sensing",                       "publisher": "MDPI",      "domain": "remote_sensing"},
    {"name": "ISPRS Journal",                        "publisher": "Elsevier",  "domain": "remote_sensing"},
    {"name": "IEEE TGRS",                            "publisher": "IEEE",      "domain": "remote_sensing"},
    {"name": "IEEE JSTARS",                          "publisher": "IEEE",      "domain": "remote_sensing"},
    {"name": "Nature Communications",               "publisher": "Nature",    "domain": "multidisciplinary"},
    {"name": "IEEE Transactions on Neural Networks", "publisher": "IEEE",      "domain": "machine_learning"},
    {"name": "Pattern Recognition",                  "publisher": "Elsevier",  "domain": "computer_vision"},
]

KEYWORDS: list[dict] = [
    # Remote sensing tasks
    {"name": "Flood Detection",            "domain": "remote_sensing"},
    {"name": "Wildfire Detection",         "domain": "remote_sensing"},
    {"name": "Crop Monitoring",            "domain": "remote_sensing"},
    {"name": "Urban Mapping",              "domain": "remote_sensing"},
    {"name": "Deforestation Detection",    "domain": "remote_sensing"},
    {"name": "Change Detection",           "domain": "remote_sensing"},
    {"name": "Land Cover Classification", "domain": "remote_sensing"},
    {"name": "Cloud Removal",             "domain": "remote_sensing"},
    {"name": "Super Resolution",          "domain": "remote_sensing"},
    {"name": "Building Extraction",       "domain": "remote_sensing"},
    {"name": "Road Extraction",           "domain": "remote_sensing"},
    {"name": "Ship Detection",            "domain": "remote_sensing"},
    {"name": "Glacier Monitoring",        "domain": "remote_sensing"},
    # Spectral indices
    {"name": "NDVI",   "domain": "remote_sensing"},
    {"name": "NDWI",   "domain": "remote_sensing"},
    {"name": "NDBI",   "domain": "remote_sensing"},
    {"name": "EVI",    "domain": "remote_sensing"},
    # ML concepts
    {"name": "Transfer Learning",         "domain": "machine_learning"},
    {"name": "Self-Supervised Learning",  "domain": "machine_learning"},
    {"name": "Few-Shot Learning",         "domain": "machine_learning"},
    {"name": "Foundation Model",          "domain": "machine_learning"},
    {"name": "Multimodal Learning",       "domain": "machine_learning"},
    {"name": "Contrastive Learning",      "domain": "machine_learning"},
    {"name": "Knowledge Distillation",    "domain": "machine_learning"},
    {"name": "Optical Imaging",          "domain": "remote_sensing"},
    {"name": "Hyperspectral Analysis",   "domain": "remote_sensing"},
    {"name": "SAR Imaging",              "domain": "remote_sensing"},
]


# ══════════════════════════════════════════════════════════════════════════════
# LOADER
# ══════════════════════════════════════════════════════════════════════════════

class OntologyLoader:
    """
    Loads the remote sensing ontology into Neo4j.

    Methods are independent — call load_all() or individual methods.
    All writes use MERGE — safe to re-run.
    """

    def __init__(self) -> None:
        from backend.graph.neo4j_client import _get_driver
        self._driver = _get_driver()

    def _run(self, cypher: str, **params) -> None:
        with self._driver.session() as s:
            s.run(cypher, **params)

    def _run_many(self, cypher: str, items: list[dict]) -> int:
        count = 0
        with self._driver.session() as s:
            for item in items:
                s.run(cypher, **item)
                count += 1
        return count

    # ── Constraints ────────────────────────────────────────────────────────────

    def create_constraints(self) -> None:
        """Add uniqueness constraints for new node types."""
        constraints = [
            "CREATE CONSTRAINT inst_name_unique   IF NOT EXISTS FOR (n:Institution) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT satellite_unique    IF NOT EXISTS FOR (n:Satellite)   REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT sensor_unique       IF NOT EXISTS FOR (n:Sensor)      REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT mission_unique      IF NOT EXISTS FOR (n:Mission)     REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT algorithm_unique    IF NOT EXISTS FOR (n:Algorithm)   REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT metric_unique       IF NOT EXISTS FOR (n:Metric)      REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT loss_unique         IF NOT EXISTS FOR (n:LossFunction)REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT conference_unique   IF NOT EXISTS FOR (n:Conference)  REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT journal_unique      IF NOT EXISTS FOR (n:Journal)     REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT keyword_unique      IF NOT EXISTS FOR (n:Keyword)     REQUIRE n.name IS UNIQUE",
        ]
        with self._driver.session() as s:
            for c in constraints:
                s.run(c)
        logger.info("[Ontology] Constraints created")

    # ── Node loaders ───────────────────────────────────────────────────────────

    def load_institutions(self) -> int:
        n = self._run_many(
            "MERGE (i:Institution {name: $name}) "
            "SET i.full_name=$full_name, i.country=$country",
            INSTITUTIONS,
        )
        logger.info(f"[Ontology] Institutions: {n}")
        return n

    def load_missions(self) -> int:
        n = 0
        with self._driver.session() as s:
            for m in MISSIONS:
                s.run(
                    "MERGE (m:Mission {name: $name}) "
                    "SET m.purpose=$purpose",
                    name=m["name"], purpose=m["purpose"],
                )
                if m.get("operator"):
                    s.run(
                        "MATCH (m:Mission {name:$mname}) "
                        "MATCH (i:Institution {name:$iname}) "
                        "MERGE (m)-[:OPERATED_BY]->(i)",
                        mname=m["name"], iname=m["operator"],
                    )
                n += 1
        logger.info(f"[Ontology] Missions: {n}")
        return n

    def load_satellites(self) -> int:
        n = 0
        with self._driver.session() as s:
            for sat in SATELLITES:
                s.run(
                    "MERGE (s:Satellite {name: $name}) "
                    "SET s.type=$type, s.resolution_m=$resolution_m",
                    name=sat["name"], type=sat["type"],
                    resolution_m=sat.get("resolution_m"),
                )
                if sat.get("mission"):
                    s.run(
                        "MATCH (s:Satellite {name:$sname}) "
                        "MATCH (m:Mission   {name:$mname}) "
                        "MERGE (s)-[:PART_OF_MISSION]->(m)",
                        sname=sat["name"], mname=sat["mission"],
                    )
                n += 1
        logger.info(f"[Ontology] Satellites: {n}")
        return n

    def load_sensors(self) -> int:
        n = self._run_many(
            "MERGE (s:Sensor {name: $name}) "
            "SET s.full_name=$full_name, s.wavelength=$wavelength",
            SENSORS,
        )
        logger.info(f"[Ontology] Sensors: {n}")
        return n

    def load_satellite_sensor_links(self) -> int:
        n = 0
        with self._driver.session() as s:
            for link in SATELLITE_SENSOR_MAP:
                s.run(
                    "MATCH (sat:Satellite {name:$sname}) "
                    "MATCH (sen:Sensor    {name:$senname}) "
                    "MERGE (sat)-[:HAS_SENSOR]->(sen)",
                    sname=link["satellite"], senname=link["sensor"],
                )
                n += 1
        logger.info(f"[Ontology] Satellite-Sensor links: {n}")
        return n

    def load_algorithms(self) -> int:
        n = self._run_many(
            "MERGE (a:Algorithm {name: $name}) "
            "SET a.type=$type, a.domain=$domain",
            ALGORITHMS,
        )
        logger.info(f"[Ontology] Algorithms: {n}")
        return n

    def load_metrics(self) -> int:
        n = self._run_many(
            "MERGE (m:Metric {name: $name}) "
            "SET m.full_name=$full_name, m.task=$task",
            METRICS,
        )
        logger.info(f"[Ontology] Metrics: {n}")
        return n

    def load_loss_functions(self) -> int:
        n = self._run_many(
            "MERGE (l:LossFunction {name: $name}) "
            "SET l.use_case=$use_case",
            LOSS_FUNCTIONS,
        )
        logger.info(f"[Ontology] Loss functions: {n}")
        return n

    def load_conferences(self) -> int:
        n = self._run_many(
            "MERGE (c:Conference {name: $name}) "
            "SET c.full_name=$full_name, c.domain=$domain",
            CONFERENCES,
        )
        logger.info(f"[Ontology] Conferences: {n}")
        return n

    def load_journals(self) -> int:
        n = self._run_many(
            "MERGE (j:Journal {name: $name}) "
            "SET j.publisher=$publisher, j.domain=$domain",
            JOURNALS,
        )
        logger.info(f"[Ontology] Journals: {n}")
        return n

    def load_keywords(self) -> int:
        n = 0
        with self._driver.session() as s:
            for kw in KEYWORDS:
                s.run(
                    "MERGE (k:Keyword {name: $name}) "
                    "SET k.domain=$domain",
                    name=kw["name"], domain=kw["domain"],
                )
                # Link keyword to its domain node if it exists
                s.run(
                    "MATCH (k:Keyword {name:$kname}) "
                    "OPTIONAL MATCH (d:Domain {key:$dkey}) "
                    "FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END | "
                    "  MERGE (k)-[:RELATED_TO]->(d))",
                    kname=kw["name"], dkey=kw["domain"],
                )
                n += 1
        logger.info(f"[Ontology] Keywords: {n}")
        return n

    # ── Load all ───────────────────────────────────────────────────────────────

    def load_all(self) -> dict[str, int]:
        """Load the complete ontology. Returns counts per category."""
        logger.info("[Ontology] Loading full remote sensing ontology...")
        self.create_constraints()

        counts = {
            "institutions":          self.load_institutions(),
            "missions":              self.load_missions(),
            "satellites":            self.load_satellites(),
            "sensors":               self.load_sensors(),
            "satellite_sensor_links":self.load_satellite_sensor_links(),
            "algorithms":            self.load_algorithms(),
            "metrics":               self.load_metrics(),
            "loss_functions":        self.load_loss_functions(),
            "conferences":           self.load_conferences(),
            "journals":              self.load_journals(),
            "keywords":              self.load_keywords(),
        }

        total = sum(counts.values())
        logger.info(f"[Ontology] Complete — {total} total entities loaded")
        return counts

    # ── Load from external file (future-proof) ─────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> "OntologyLoader":
        """
        Load ontology from an external JSON file.

        Expected JSON structure:
        {
            "institutions": [...],
            "satellites":   [...],
            ...
        }

        This allows curated ontology datasets to be ingested
        without modifying this file.
        """
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        loader = cls()

        # Dynamically update module-level dicts and reload
        if "institutions" in data:
            INSTITUTIONS.clear()
            INSTITUTIONS.extend(data["institutions"])
        if "satellites" in data:
            SATELLITES.clear()
            SATELLITES.extend(data["satellites"])
        # ... extend for other types as needed

        return loader


# ══════════════════════════════════════════════════════════════════════════════
# STATS + CLI
# ══════════════════════════════════════════════════════════════════════════════

def verify() -> None:
    from backend.graph.neo4j_client import _get_driver

    driver = _get_driver()
    node_types = [
        "Institution", "Mission", "Satellite", "Sensor",
        "Algorithm", "Metric", "LossFunction",
        "Conference", "Journal", "Keyword",
    ]

    print("\nOntology verification:")
    with driver.session() as s:
        for ntype in node_types:
            result = s.run(
                f"MATCH (n:{ntype}) RETURN count(n) AS n"
            ).single()
            count = result["n"] if result else 0
            print(f"  {ntype:<15} : {count}")

        rels = s.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS n "
            "ORDER BY n DESC"
        ).data()
        print("\nRelationships:")
        for row in rels[:10]:
            print(f"  {row['t']:<25} : {row['n']}")


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\n" + "="*55)
    print("ASTRONEXUS — REMOTE SENSING ONTOLOGY LOADER")
    print("="*55)

    try:
        loader = OntologyLoader()
        counts = loader.load_all()

        print("\nLoaded:")
        for category, n in counts.items():
            print(f"  {category:<25} {n}")

        verify()
        print("\n✓ Ontology loaded successfully.")
        print("  Papers ingested via POST /upload will now link to")
        print("  satellites, sensors, algorithms, metrics and more.")

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        print("  Ensure Neo4j is running: docker start astronexus-neo4j")
        sys.exit(1)


if __name__ == "__main__":
    run()