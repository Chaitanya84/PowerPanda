"""
Step 1: Extract entities and relations from documents using LLM (OpenAI GPT-4).
Production mode — no rule-based fallback.

© 2026 Chaitanya Priya. All rights reserved. PowerPanda™
"""

import logging

logger = logging.getLogger(__name__)
import re
import os
import json
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


@dataclass
class Entity:
    name: str
    entity_type: str


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str


def _get_openai_client(api_key: str = None) -> OpenAI:
    """Create OpenAI client."""
    key = api_key or OPENAI_API_KEY
    return OpenAI(api_key=key)


def _extract_with_llm(text: str, api_key: str = None) -> tuple[list[Entity], list[Relation]]:
    """Extract entities and relations using OpenAI GPT-4."""
    prompt = (
        "Extract all entities and relations from the following text.\n"
        "Return a JSON object with two keys:\n"
        '- "entities": list of objects with "name" and "entity_type" '
        "(PERSON, ORGANIZATION, PRODUCT, LOCATION, CONCEPT, TECHNOLOGY, EVENT, etc.)\n"
        '- "relations": list of objects with "source", "target", and "relation_type"\n'
        "Be thorough — extract ALL meaningful entities and relationships.\n"
        "Return ONLY valid JSON, no other text.\n\n"
        f"TEXT:\n{text}"
    )

    logger.info("\n" + "="*60)
    logger.info("[EXTRACT] Sending text to OpenAI for entity/relation extraction...")
    logger.info(f"[EXTRACT] Text preview (first 200 chars): {text[:200]}...")
    logger.info("="*60)

    client = _get_openai_client(api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert at extracting structured knowledge from text. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content

    logger.info(f"\n[EXTRACT] Raw LLM response (first 500 chars):\n{answer[:500]}")

    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*(.*?)```', answer, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            raise ValueError(f"Failed to parse LLM response as JSON: {answer[:200]}")

    entities = [Entity(name=e["name"], entity_type=e.get("entity_type", "UNKNOWN")) for e in data.get("entities", [])]
    relations = [Relation(source=r["source"], target=r["target"], relation_type=r["relation_type"]) for r in data.get("relations", [])]

    logger.info(f"\n[EXTRACT] === ENTITIES FOUND ({len(entities)}) ===")
    for e in entities:
        logger.info(f"  • {e.name} [{e.entity_type}]")

    logger.info(f"\n[EXTRACT] === RELATIONS FOUND ({len(relations)}) ===")
    for r in relations:
        logger.info(f"  • {r.source} --[{r.relation_type}]--> {r.target}")

    return entities, relations


def extract_from_documents(documents: list[str], api_key: str = None) -> tuple[list[Entity], list[Relation]]:
    """Process all documents and return entities + relations using LLM extraction."""
    all_entities = []
    all_relations = []

    logger.info("\n" + "#"*60)
    logger.info(f"[EXTRACT] Starting extraction for {len(documents)} document(s)")
    logger.info("#"*60)

    for i, doc in enumerate(documents):
        logger.info(f"\n[EXTRACT] --- Processing Document {i+1}/{len(documents)} ---")
        entities, relations = _extract_with_llm(doc, api_key)
        all_entities.extend(entities)
        all_relations.extend(relations)

    # Deduplicate entities by name
    seen = set()
    unique_entities = []
    for e in all_entities:
        key = e.name.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)

    logger.info(f"\n[EXTRACT] === FINAL SUMMARY ===")
    logger.info(f"[EXTRACT] Total entities (before dedup): {len(all_entities)}")
    logger.info(f"[EXTRACT] Unique entities (after dedup): {len(unique_entities)}")
    logger.info(f"[EXTRACT] Total relations: {len(all_relations)}")

    return unique_entities, all_relations
