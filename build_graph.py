"""
Step 2: Build a knowledge graph using in-memory storage + Mermaid visualization.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from extract_graph import Entity, Relation

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GRAPH_STORE_DIR = Path("powerpanda_store")
GRAPH_STORE_DIR.mkdir(exist_ok=True)


class InMemoryGraph:
    """Simple in-memory knowledge graph with adjacency list."""

    def __init__(self):
        self.nodes: dict[str, str] = {}  # name -> entity_type
        self.edges: list[tuple[str, str, str]] = []  # (source, target, relation_type)

    def add_node(self, name: str, entity_type: str):
        self.nodes[name] = entity_type

    def add_edge(self, source: str, target: str, relation_type: str):
        self.edges.append((source, target, relation_type))

    def get_neighbors(self, node: str, hops: int = 1) -> set[str]:
        """Get all nodes reachable within N hops."""
        visited = {node}
        frontier = {node}
        for hop in range(hops):
            next_frontier = set()
            for n in frontier:
                for src, tgt, _ in self.edges:
                    if src == n and tgt not in visited:
                        next_frontier.add(tgt)
                    elif tgt == n and src not in visited:
                        next_frontier.add(src)
            visited.update(next_frontier)
            frontier = next_frontier
            logger.info(f"    [GRAPH] Hop {hop+1} from '{node}': found neighbors {next_frontier}")
        return visited - {node}

    def get_node_context(self, node: str, hops: int = 2) -> str:
        """Get text context around a node by traversing N hops."""
        logger.info(f"\n  [GRAPH] Getting context for node: '{node}' (hops={hops})")
        reachable = self.get_neighbors(node, hops)
        reachable.add(node)
        logger.info(f"  [GRAPH] Reachable nodes: {reachable}")
        context_lines = []
        for src, tgt, rel in self.edges:
            if src in reachable or tgt in reachable:
                context_lines.append(f"{src} --[{rel}]--> {tgt}")
        logger.info(f"  [GRAPH] Context lines collected: {len(context_lines)}")
        for line in context_lines:
            logger.info(f"    {line}")
        return "\n".join(context_lines)

    def to_mermaid(self, highlight_nodes: list[str] = None) -> str:
        """Generate Mermaid graph code."""
        lines = ["graph LR"]
        node_ids = {}
        for i, name in enumerate(self.nodes.keys()):
            node_ids[name] = f"N{i}"
        for name, entity_type in self.nodes.items():
            nid = node_ids[name]
            label = f"{name}\\n({entity_type})"
            lines.append(f"    {nid}[\"{label}\"]")
        for src, tgt, rel in self.edges:
            if src in node_ids and tgt in node_ids:
                safe_rel = rel.replace("_", " ")
                lines.append(f"    {node_ids[src]} -->|\"{safe_rel}\"| {node_ids[tgt]}")
        if highlight_nodes:
            highlighted_ids = [node_ids[n] for n in highlight_nodes if n in node_ids]
            if highlighted_ids:
                for hid in highlighted_ids:
                    lines.append(f"    style {hid} fill:#ff9,stroke:#f90,stroke-width:3px")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [{"source": s, "target": t, "relation": r} for s, t, r in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InMemoryGraph":
        graph = cls()
        graph.nodes = data.get("nodes", {})
        graph.edges = [(e["source"], e["target"], e["relation"]) for e in data.get("edges", [])]
        return graph

    def save(self, graph_name: str = "powerpanda"):
        path = GRAPH_STORE_DIR / f"{graph_name}_graph.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        # Save mermaid diagram to disk
        self.save_mermaid(graph_name)

    def save_mermaid(self, graph_name: str = "powerpanda", highlight_nodes: list[str] = None):
        """Save the Mermaid diagram to a .md file on disk."""
        mermaid_code = self.to_mermaid(highlight_nodes=highlight_nodes)
        mermaid_path = GRAPH_STORE_DIR / f"{graph_name}_mermaid.md"
        mermaid_path.write_text(f"```mermaid\n{mermaid_code}\n```\n", encoding="utf-8")
        logger.info(f"[GRAPH] Mermaid diagram saved to: {mermaid_path}")

    @classmethod
    def load(cls, graph_name: str = "powerpanda") -> "InMemoryGraph":
        path = GRAPH_STORE_DIR / f"{graph_name}_graph.json"
        if path.exists():
            return cls.from_dict(json.loads(path.read_text()))
        return cls()


def get_graph(graph_name: str = "powerpanda") -> InMemoryGraph:
    return InMemoryGraph.load(graph_name)


def build_knowledge_graph(entities: list[Entity], relations: list[Relation], graph_name: str = "powerpanda") -> InMemoryGraph:
    """Build the knowledge graph from entities and relations."""
    logger.info("\n" + "="*60)
    logger.info("[GRAPH] Building knowledge graph...")
    logger.info("="*60)

    graph = InMemoryGraph()

    for entity in entities:
        graph.add_node(entity.name, entity.entity_type)
        logger.info(f"  [GRAPH] Added node: {entity.name} [{entity.entity_type}]")

    for rel in relations:
        if rel.source not in graph.nodes:
            graph.add_node(rel.source, "UNKNOWN")
            logger.info(f"  [GRAPH] Added missing node: {rel.source} [UNKNOWN]")
        if rel.target not in graph.nodes:
            graph.add_node(rel.target, "UNKNOWN")
            logger.info(f"  [GRAPH] Added missing node: {rel.target} [UNKNOWN]")
        graph.add_edge(rel.source, rel.target, rel.relation_type)
        logger.info(f"  [GRAPH] Added edge: {rel.source} --[{rel.relation_type}]--> {rel.target}")

    graph.save(graph_name)
    logger.info(f"\n[GRAPH] Graph saved! Total: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph


def get_node_context(graph: InMemoryGraph, node: str, hops: int = 2) -> str:
    if graph is None:
        return ""
    return graph.get_node_context(node, hops)


def get_all_node_names(graph: InMemoryGraph) -> list[str]:
    if graph is None:
        return []
    return list(graph.nodes.keys())


def find_relevant_nodes_by_embedding(
    query: str,
    graph: InMemoryGraph,
    api_key: str,
    top_k: int = 5,
    node_embeddings_cache: dict = None,
) -> list[str]:
    """Find relevant graph nodes using embedding similarity (text-embedding-ada-002)."""
    if graph is None:
        return []
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=api_key or OPENAI_API_KEY,
    )

    node_names = get_all_node_names(graph)
    if not node_names:
        return []

    logger.info(f"\n[SEARCH] Embedding query: '{query}'")
    logger.info(f"[SEARCH] Comparing against {len(node_names)} graph nodes:")
    for n in node_names:
        logger.info(f"  • {n}")

    # Embed node names (use cache if available)
    if (node_embeddings_cache and "vectors" in node_embeddings_cache
            and node_embeddings_cache.get("names") == node_names):
        node_vectors = node_embeddings_cache["vectors"]
        logger.info(f"[SEARCH] Using cached node embeddings")
    else:
        logger.info(f"[SEARCH] Computing embeddings for all nodes...")
        node_vectors = embeddings_model.embed_documents(node_names)
        if node_embeddings_cache is not None:
            node_embeddings_cache["vectors"] = node_vectors
            node_embeddings_cache["names"] = node_names

    # Embed query
    query_vector = embeddings_model.embed_query(query)

    # Cosine similarity
    node_matrix = np.array(node_vectors)
    query_vec = np.array(query_vector)
    similarities = np.dot(node_matrix, query_vec) / (
        np.linalg.norm(node_matrix, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )

    # Show all similarity scores
    logger.info(f"\n[SEARCH] === SIMILARITY SCORES ===")
    scored = sorted(zip(node_names, similarities), key=lambda x: -x[1])
    for name, score in scored:
        marker = " ✓" if score > 0.3 else ""
        logger.info(f"  {score:.4f}  {name}{marker}")

    # Get top-K above threshold
    top_indices = np.argsort(similarities)[::-1][:top_k]
    relevant = [node_names[i] for i in top_indices if similarities[i] > 0.3]

    logger.info(f"\n[SEARCH] Selected top-{top_k} nodes (threshold > 0.3): {relevant}")
    return relevant
