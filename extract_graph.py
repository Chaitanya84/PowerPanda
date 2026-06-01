"""
Step 1: Extract entities and relations from documents using OpenAI.
GraphRAG-optimized extraction.

© 2026 Chaitanya Priya. All rights reserved. PowerPanda™
"""

import json
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ------------------------------------------------------------
# DATA MODELS
# ------------------------------------------------------------

@dataclass
class Entity:
    name: str
    entity_type: str


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str
    confidence: float = 1.0
    evidence: str = ""


# ------------------------------------------------------------
# ONTOLOGY
# ------------------------------------------------------------

ALLOWED_RELATIONS = {
    "CAUSES",
    "RESULTS_IN",
    "DEPENDS_ON",
    "USES",
    "IMPLEMENTS",
    "CONTAINS",
    "PART_OF",
    "OWNED_BY",
    "PRODUCES",
    "CONSUMES",
    "MEASURES",
    "IMPACTS",
    "PREVENTS",
    "TRIGGERS",
    "REQUIRES",
    "SUPPORTS",
    "GENERATES",
    "DERIVED_FROM",
}

ENTITY_TYPES = {
    "PERSON",
    "ORGANIZATION",
    "PRODUCT",
    "TECHNOLOGY",
    "PROCESS",
    "BUSINESS_CONCEPT",
    "METRIC",
    "EVENT",
    "RISK",
    "OUTCOME",
    "SYSTEM",
    "DATASET",
    "DOCUMENT",
    "LOCATION",
}


# ------------------------------------------------------------
# CLIENT
# ------------------------------------------------------------

def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize_entity_name(name: str) -> str:
    """
    Canonicalize entity names to reduce duplicates.
    """

    if not name:
        return ""

    name = name.strip()

    aliases = {
        "GPT4": "GPT-4",
        "GPT 4": "GPT-4",
        "GPT4o": "GPT-4o",
        "OpenAI Inc.": "OpenAI",
        "OpenAI LLC": "OpenAI",
    }

    return aliases.get(name, name)


# ------------------------------------------------------------
# EXTRACTION
# ------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert knowledge graph construction system for GraphRAG.

Your goal is to create a retrieval-optimized knowledge graph.

=========================
ENTITY EXTRACTION RULES
=========================

Extract entities that maximize future retrieval quality.

Prioritize:

- Named entities
- Organizations
- Products
- Technologies
- Systems
- Technical components
- Business concepts
- Metrics
- Processes
- Events
- Risks
- Outcomes
- Decisions
- Root causes

Avoid:

- Pronouns
- Generic nouns
- Vague concepts
- Low-information entities

BAD:
Customer
Data
Service
System
Revenue

GOOD:
Customer Churn
Revenue Recognition
Subscription Billing
Failed Payment Event
Fraud Detection Model
Webhook Processing

Entity names must be understandable in isolation.

Use canonical names whenever possible.

Allowed entity types:

PERSON
ORGANIZATION
PRODUCT
TECHNOLOGY
PROCESS
BUSINESS_CONCEPT
METRIC
EVENT
RISK
OUTCOME
SYSTEM
DATASET
DOCUMENT
LOCATION

=========================
RELATION EXTRACTION RULES
=========================

Use ONLY these relation types:

CAUSES
RESULTS_IN
DEPENDS_ON
USES
IMPLEMENTS
CONTAINS
PART_OF
OWNED_BY
PRODUCES
CONSUMES
MEASURES
IMPACTS
PREVENTS
TRIGGERS
REQUIRES
SUPPORTS
GENERATES
DERIVED_FROM

DO NOT USE:

related_to
associated_with
linked_to
mentions
references

Choose the most specific relation.

For each relation provide:

- source
- target
- relation_type
- confidence
- evidence

confidence must be between 0 and 1.

Only emit relations with confidence >= 0.70.

=========================
OUTPUT FORMAT
=========================

Return ONLY valid JSON.

{
  "entities": [
    {
      "name": "...",
      "entity_type": "..."
    }
  ],
  "relations": [
    {
      "source": "...",
      "target": "...",
      "relation_type": "...",
      "confidence": 0.92,
      "evidence": "text supporting relation"
    }
  ]
}
"""


def _extract_with_llm(text: str) -> tuple[list[Entity], list[Relation]]:
    """
    Extract entities and relations using OpenAI.
    """

    prompt = f"""
Extract a retrieval-optimized knowledge graph from the text.

Requirements:

1. Use canonical entity names.
2. Prefer descriptive multi-word entities.
3. Merge aliases when obvious.
4. Avoid duplicate concepts.
5. Extract causal relationships whenever possible.
6. Extract process relationships.
7. Extract dependency relationships.
8. Only create relationships directly supported by the text.
9. Use only approved relation types.

TEXT:

{text}
"""

    logger.info("\n" + "=" * 80)
    logger.info("[EXTRACT] Sending document to OpenAI")
    logger.info("=" * 80)

    client = _get_openai_client()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response.choices[0].message.content

    logger.info(
        f"[EXTRACT] Raw response preview:\n{answer[:1000]}"
    )

    data = json.loads(answer)

    entities = []

    for e in data.get("entities", []):

        name = normalize_entity_name(e.get("name", "").strip())

        if not name:
            continue

        entity_type = e.get("entity_type", "BUSINESS_CONCEPT")

        if entity_type not in ENTITY_TYPES:
            entity_type = "BUSINESS_CONCEPT"

        entities.append(
            Entity(
                name=name,
                entity_type=entity_type,
            )
        )

    relations = []

    for r in data.get("relations", []):

        confidence = float(r.get("confidence", 0))

        if confidence < 0.70:
            continue

        relation_type = r.get("relation_type", "").upper()

        if relation_type not in ALLOWED_RELATIONS:
            continue

        source = normalize_entity_name(r.get("source", ""))
        target = normalize_entity_name(r.get("target", ""))

        if not source or not target:
            continue

        relations.append(
            Relation(
                source=source,
                target=target,
                relation_type=relation_type,
                confidence=confidence,
                evidence=r.get("evidence", ""),
            )
        )

    logger.info(f"[EXTRACT] Entities extracted: {len(entities)}")
    logger.info(f"[EXTRACT] Relations extracted: {len(relations)}")

    logger.info("\n[EXTRACT] === ENTITIES ===")
    for e in entities:
        logger.info(f"  • {e.name} [{e.entity_type}]")

    logger.info("\n[EXTRACT] === RELATIONS ===")
    for r in relations:
        logger.info(
            f"  • {r.source} --[{r.relation_type}]--> {r.target} "
            f"(confidence={r.confidence:.2f})"
        )

    return entities, relations


# ------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------

def extract_from_documents(
    documents: list[str],
) -> tuple[list[Entity], list[Relation]]:
    """
    Process documents and build graph entities + relations.
    """

    logger.info("\n" + "#" * 80)
    logger.info(
        f"[EXTRACT] Processing {len(documents)} document(s)"
    )
    logger.info("#" * 80)

    all_entities = []
    all_relations = []

    for idx, doc in enumerate(documents):

        logger.info(
            f"\n[EXTRACT] Document {idx + 1}/{len(documents)}"
        )

        entities, relations = _extract_with_llm(doc)

        all_entities.extend(entities)
        all_relations.extend(relations)

    # --------------------------------------------------------
    # ENTITY DEDUP
    # --------------------------------------------------------

    entity_map = {}

    for e in all_entities:
        key = e.name.lower().strip()

        if key not in entity_map:
            entity_map[key] = e

    unique_entities = list(entity_map.values())

    # --------------------------------------------------------
    # RELATION DEDUP
    # --------------------------------------------------------

    relation_seen = set()
    unique_relations = []

    for r in all_relations:

        key = (
            r.source.lower(),
            r.target.lower(),
            r.relation_type,
        )

        if key not in relation_seen:
            relation_seen.add(key)
            unique_relations.append(r)

    logger.info("\n" + "=" * 80)
    logger.info("[EXTRACT] FINAL SUMMARY")
    logger.info("=" * 80)

    logger.info(
        f"Entities before dedup: {len(all_entities)}"
    )

    logger.info(
        f"Entities after dedup: {len(unique_entities)}"
    )

    logger.info(
        f"Relations before dedup: {len(all_relations)}"
    )

    logger.info(
        f"Relations after dedup: {len(unique_relations)}"
    )

    return unique_entities, unique_relations