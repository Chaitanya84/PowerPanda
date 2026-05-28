"""
Step 3: PowerPanda Query Engine (Production)

Full pipeline:
1. Retrieves graph context via embedding similarity on in-memory graph nodes
2. Retrieves document context via FAISS vector search
3. Generates answer using OpenAI GPT-4

© 2026 Chaitanya Priya. All rights reserved. PowerPanda™
"""
import logging
logger = logging.getLogger(__name__)

import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from build_graph import (
    get_graph,
    get_node_context,
    find_relevant_nodes_by_embedding,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


def build_powerpanda_prompt(query: str, graph_context: str, doc_context: str) -> str:
    """Construct an augmented prompt combining graph context and document context."""
    return f"""Answer the following question using ONLY the provided context.

=== KNOWLEDGE GRAPH CONTEXT ===
{graph_context if graph_context else "(no graph context found)"}

=== RELEVANT DOCUMENTS ===
{doc_context if doc_context else "(no documents found)"}

=== QUESTION ===
{query}

Provide a clear, comprehensive answer based on the context above."""


from pathlib import Path

SOUL_MD_PATH = Path(__file__).parent / ".github" / "copilot" / "soul.md"


def _load_system_prompt() -> str:
    """Load the system prompt dynamically from soul.md."""
    if SOUL_MD_PATH.exists():
        content = SOUL_MD_PATH.read_text(encoding="utf-8")
        stripped_lines = content.strip().splitlines()
        if stripped_lines and stripped_lines[0].startswith("# "):
            stripped_lines = stripped_lines[1:]
        return "\n".join(stripped_lines).strip()
    logger.warning("[ANSWER] soul.md not found — using default system prompt")
    return "You are a helpful, friendly assistant with a sharp sense of humor."


def generate_answer(prompt: str, api_key: str) -> str:
    """Generate answer using OpenAI GPT-4."""
    logger.info("\n" + "="*60)
    logger.info("[ANSWER] Sending augmented prompt to OpenAI GPT-4...")
    logger.info(f"[ANSWER] Prompt length: {len(prompt)} characters")
    logger.info("="*60)
    logger.info(f"\n[ANSWER] === FULL PROMPT SENT TO LLM ===\n{prompt}\n{'='*60}")

    client = OpenAI(api_key=api_key or OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _load_system_prompt()},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    answer = response.choices[0].message.content
    logger.info(f"\n[ANSWER] === LLM RESPONSE ===\n{answer}\n{'='*60}")
    return answer


def query_powerpanda(
    query: str,
    api_key: str,
    vectorstore: FAISS = None,
    graph_name: str = "powerpanda",
    node_embeddings_cache: dict = None,
    top_k_docs: int = 4,
    top_k_nodes: int = 5,
    hops: int = 2,
) -> tuple[str, list, str, list]:
    """
    Full PowerPanda query pipeline.
    Returns: (answer, source_docs, graph_context, relevant_nodes)
    """
    logger.info("\n" + "#"*60)
    logger.info(f"[QUERY] === POWERPANDA PIPELINE START ===")
    logger.info(f"[QUERY] Question: '{query}'")
    logger.info(f"[QUERY] Settings: top_k_docs={top_k_docs}, top_k_nodes={top_k_nodes}, hops={hops}")
    logger.info("#"*60)

    # Step 1: Find relevant nodes via embedding similarity
    logger.info("\n[QUERY] STEP 1: Finding relevant graph nodes via embedding similarity...")
    graph_context = ""
    relevant_nodes = []
    try:
        graph = get_graph(graph_name)
        relevant_nodes = find_relevant_nodes_by_embedding(
            query, graph, api_key, top_k=top_k_nodes,
            node_embeddings_cache=node_embeddings_cache,
        )

        # Step 2: Get graph context (subgraph traversal)
        logger.info(f"\n[QUERY] STEP 2: Traversing graph from relevant nodes...")
        graph_context_parts = []
        for node in relevant_nodes:
            ctx = get_node_context(graph, node, hops=hops)
            if ctx:
                graph_context_parts.append(f"[{node}]\n{ctx}")
        graph_context = "\n\n".join(graph_context_parts)

        logger.info(f"\n[QUERY] === FINAL GRAPH CONTEXT ===")
        if graph_context:
            logger.info(graph_context)
        else:
            logger.info("  (empty - no graph context found)")
    except Exception as e:
        graph_context = f"(Graph error: {e})"
        logger.info(f"[QUERY] Graph error: {e}")

    # Step 3: Document retrieval via FAISS
    logger.info(f"\n[QUERY] STEP 3: Retrieving documents from FAISS (top_k={top_k_docs})...")
    doc_context = ""
    source_docs = []
    if vectorstore:
        source_docs = vectorstore.similarity_search(query, k=top_k_docs)
        doc_context = "\n\n---\n\n".join(doc.page_content for doc in source_docs)
        logger.info(f"[QUERY] Found {len(source_docs)} relevant document chunks:")
        for i, doc in enumerate(source_docs):
            src = doc.metadata.get("source_file", "unknown")
            logger.info(f"  Chunk {i+1} (from {src}): {doc.page_content[:100]}...")
    else:
        logger.info("[QUERY] No vectorstore available!")

    # Step 4: Build augmented prompt and generate answer
    logger.info(f"\n[QUERY] STEP 4: Building augmented prompt and generating answer...")
    prompt = build_powerpanda_prompt(query, graph_context, doc_context)
    answer = generate_answer(prompt, api_key)

    logger.info(f"\n[QUERY] === PIPELINE COMPLETE ===\n")
    return answer, source_docs, graph_context, relevant_nodes
