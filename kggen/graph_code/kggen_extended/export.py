"""Export utilities for ExtendedGraph.

Produces:
  - GraphML with passage nodes + Source edges (mirrors AutoSchemaKG)
  - triples.csv / triples.jsonl with passage_ids column
  - passages.csv / passages.jsonl registry
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from kggen_extended.models import ExtendedGraph, _triple_to_key


def _relation_edge_key(predicate: str) -> str:
    return f"Relation::{predicate}"


def _source_edge_key(passage_id: str) -> str:
    return f"Source::{passage_id}"


def _get_relation_passage_ids(
    graph: ExtendedGraph,
    subject: str,
    predicate: str,
    obj: str,
) -> List[str]:
    """Return strict triplet-level provenance for a relation edge.

    Relation edges must only inherit passages from `triplet_passages`.
    We intentionally do not derive relation provenance from entity-level
    mentions, because an entity can appear in a passage without the relation
    itself being stated there.
    """
    if not graph.triplet_passages:
        return []

    triple_key = _triple_to_key((subject, predicate, obj))
    return sorted(graph.triplet_passages.get(triple_key, set()))


def to_nx_extended(graph: ExtendedGraph) -> nx.MultiDiGraph:
    """Convert ExtendedGraph to a networkx MultiDiGraph with passage nodes and Source edges.

    Node types:
      - entity: entities from the graph
      - passage: source passages with text and metadata attributes

    Edge types:
      - Relation: triplet (entity → entity, attribute 'relation')
      - Source: mention_in (entity → passage)
    """
    G = nx.MultiDiGraph()

    # ── Add entity nodes ──
    for entity in graph.entities:
        G.add_node(entity, type="entity")

    # ── Add passage nodes ──
    for pid, pinfo in graph.passages.items():
        G.add_node(
            pid,
            type="passage",
            text=pinfo.get("text", ""),
            metadata=json.dumps(pinfo.get("metadata", {}), ensure_ascii=False)
            if pinfo.get("metadata")
            else "{}",
        )

    # ── Add Source edges: entity → passage ──
    if graph.entity_passages:
        for entity, pids in graph.entity_passages.items():
            if entity in G.nodes:  # Only if entity exists in graph
                for pid in pids:
                    if pid in G.nodes:  # Only if passage exists in graph
                        G.add_edge(
                            entity,
                            pid,
                            key=_source_edge_key(pid),
                            type="Source",
                            relation="mention in",
                        )

    # ── Add Relation edges ──
    for subject, predicate, obj in graph.relations:
        relation_pids = _get_relation_passage_ids(graph, subject, predicate, obj)
        if not relation_pids:
            continue
        G.add_edge(
            subject,
            obj,
            key=_relation_edge_key(predicate),
            type="Relation",
            relation=predicate,
        )

    return G


def to_nx_extended_with_passage_ids(
    graph: ExtendedGraph,
) -> nx.MultiDiGraph:
    """Like to_nx_extended() but also adds passage_ids attribute to Relation edges.

    The passage_ids attribute on a Relation edge comes strictly from
    `triplet_passages` for that exact (subject, predicate, object) triple.
    We do not derive relation provenance from entity-level mentions.
    """
    G = to_nx_extended(graph)

    # Add passage_ids to Relation edges
    for subject, predicate, obj in graph.relations:
        relation_pids = _get_relation_passage_ids(graph, subject, predicate, obj)
        if not relation_pids:
            continue

        edge_key = _relation_edge_key(predicate)
        if G.has_edge(subject, obj, edge_key):
            G.edges[subject, obj, edge_key]["passage_ids"] = ",".join(relation_pids)

    return G


def export_graphml(
    graph: ExtendedGraph,
    output_path: str,
    include_passage_ids_on_edges: bool = True,
):
    """Export ExtendedGraph to GraphML file.

    Args:
        graph: The ExtendedGraph to export.
        output_path: Path for the .graphml file.
        include_passage_ids_on_edges: If True, add passage_ids attribute
            to Relation edges from strict triplet-level provenance.
    """
    if include_passage_ids_on_edges:
        G = to_nx_extended_with_passage_ids(graph)
    else:
        G = to_nx_extended(graph)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, output_path)

    nn = G.number_of_nodes()
    ne = G.number_of_edges()
    n_passage = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "passage")
    n_source = sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "Source")
    n_relation = sum(1 for _, _, d in G.edges(data=True) if d.get("type") == "Relation")

    print(f"  GraphML: {output_path}  ({nn} nodes [{n_passage} passages], "
          f"{ne} edges [{n_source} Source, {n_relation} Relation])")


def export_triples(
    graph: ExtendedGraph,
    output_dir: str,
    prefix: str = "triples",
):
    """Export triplets as CSV and JSONL with passage_ids column.

    Produces:
      - {prefix}.csv
      - {prefix}.jsonl

    The passage_ids column contains comma-joined sorted UUIDs.
    For triplets after dedup, this will contain multiple IDs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build triples data
    triples_data: List[Dict[str, Any]] = []

    for subject, predicate, obj in graph.relations:
        relation_pids = _get_relation_passage_ids(graph, subject, predicate, obj)
        if not relation_pids:
            continue

        row: Dict[str, Any] = {
            "source": subject,
            "relation": predicate,
            "target": obj,
            "passage_ids": "",
        }

        # Strict triplet-level provenance only.
        row["passage_ids"] = ",".join(relation_pids)

        triples_data.append(row)

    # Write CSV
    df = pd.DataFrame(triples_data, columns=["source", "relation", "target", "passage_ids"])
    csv_path = out_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV:    {csv_path}")

    # Write JSONL
    jsonl_path = out_dir / f"{prefix}.jsonl"
    df.to_json(jsonl_path, orient="records", lines=True)
    print(f"  JSONL:  {jsonl_path}")

    # Stats
    print(f"  Total triples: {len(df)}")
    print(f"  Unique sources: {df['source'].nunique()}, "
          f"targets: {df['target'].nunique()}, "
          f"relations: {df['relation'].nunique()}")

    # Multi-passage stats
    has_passage = df[df["passage_ids"].str.len() > 0]
    multi = df[df["passage_ids"].str.contains(",", na=False)]
    print(f"  Triplets with passage provenance: {len(has_passage)}")
    if len(multi):
        print(f"  Triplets with multiple passages: {len(multi)}")
        for _, multi_row in multi.head(5).iterrows():
            print(f"    {multi_row['source']} → [{multi_row['relation']}] → {multi_row['target']}  "
                  f"[passages: {multi_row['passage_ids']}]")


def export_passages(
    graph: ExtendedGraph,
    output_dir: str,
    prefix: str = "passages",
):
    """Export the passage registry as CSV and JSON.

    Produces:
      - {prefix}.csv  (passage_id, text, metadata_json)
      - {prefix}.json (full passage registry as JSON)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not graph.passages:
        print("  No passages to export.")
        return

    # Write CSV
    rows = []
    for pid, pinfo in graph.passages.items():
        metadata_str = json.dumps(pinfo.get("metadata", {}), ensure_ascii=False)
        rows.append({
            "passage_id": pid,
            "text": pinfo.get("text", ""),
            "metadata": metadata_str,
        })

    df = pd.DataFrame(rows, columns=["passage_id", "text", "metadata"])
    csv_path = out_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  CSV:    {csv_path}  ({len(df)} passages)")

    # Write JSON
    json_path = out_dir / f"{prefix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph.passages, f, indent=2, ensure_ascii=False)
    print(f"  JSON:   {json_path}")


def export_all(
    graph: ExtendedGraph,
    output_dir: str,
    include_passage_ids_on_edges: bool = True,
):
    """Export all formats: GraphML, triples CSV/JSONL, passages CSV/JSON.

    Args:
        graph: The ExtendedGraph to export.
        output_dir: Directory to write output files.
        include_passage_ids_on_edges: If True, add passage_ids to GraphML Relation edges.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Exporting ExtendedGraph ...")
    print("=" * 60)

    # GraphML
    export_graphml(
        graph,
        str(out_dir / "aggregated_graph.graphml"),
        include_passage_ids_on_edges=include_passage_ids_on_edges,
    )

    # Triples
    export_triples(graph, str(out_dir))

    # Passages
    export_passages(graph, str(out_dir))

    print("=" * 60)
