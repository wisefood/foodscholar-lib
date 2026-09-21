"""High fidelity visualization utilities for kggen-extended knowledge graphs."""

import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List
import colorsys
import webbrowser

from kggen_extended.models import ExtendedGraph


def _string_to_color(label: str) -> str:
    """Generate a deterministic pastel-like color for a given label."""
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()
    hue = int(digest[:2], 16) / 255.0
    saturation = 0.55 + (int(digest[2:4], 16) / 255.0) * 0.3
    lightness = 0.45 + (int(digest[4:6], 16) / 255.0) * 0.25
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _sorted_ignore_case(items: Iterable[str]) -> List[str]:
    return sorted(items, key=lambda value: value.lower())


def visualize_kg(graph: ExtendedGraph, output_path: str, open_in_browser: bool = False):
    """Visualize an ExtendedGraph and save to HTML file."""
    view_model = _build_view_model(graph)

    template_path = Path(__file__).parent.parent / "templates" / "visualize.html"
    if template_path.exists():
        html_template = template_path.read_text()
    else:
        html_template = """<!DOCTYPE html>
<html>
<head><title>Knowledge Graph</title></head>
<body>
  <div id="graph"></div>
  <script type="application/json" id="graph-data">__GRAPH_DATA__</script>
</body>
</html>"""

    html = html_template.replace("__GRAPH_DATA__", json.dumps(view_model))

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")

    if open_in_browser:
        webbrowser.open(str(output_path.resolve()))


def _build_view_model(graph: ExtendedGraph) -> dict[str, Any]:
    # Collect all entities from both the entities set and relations
    all_entities = set(graph.entities)
    for subject, _, obj in graph.relations:
        all_entities.add(subject)
        all_entities.add(obj)
    entities = _sorted_ignore_case(all_entities)

    relations = sorted(
        graph.relations,
        key=lambda triple: (triple[1].lower(), triple[0].lower(), triple[2].lower()),
    )

    entity_clusters = graph.entity_clusters or {}
    edge_clusters = graph.edge_clusters or {}

    entity_member_to_cluster: dict[str, str] = {}
    cluster_view: List[Dict[str, Any]] = []

    for representative, members in entity_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"entity::{representative}")
        cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            entity_member_to_cluster[member] = representative

    node_color_lookup: dict[str, str] = {}
    if cluster_view:
        for cluster in cluster_view:
            for member in cluster["members"]:
                node_color_lookup[member] = cluster["color"]
    else:
        for entity in entities:
            node_color_lookup[entity] = _string_to_color(f"entity::{entity}")

    edge_member_to_cluster: dict[str, str] = {}
    edge_cluster_view: List[Dict[str, Any]] = []

    for representative, members in edge_clusters.items():
        full_members = set(members)
        full_members.add(representative)
        ordered_members = _sorted_ignore_case(full_members)
        color = _string_to_color(f"edge::{representative}")
        edge_cluster_view.append(
            {
                "id": representative,
                "label": representative,
                "members": ordered_members,
                "size": len(ordered_members),
                "color": color,
            }
        )
        for member in ordered_members:
            edge_member_to_cluster[member] = representative

    degree = Counter()
    indegree = Counter()
    outdegree = Counter()
    predicate_counts = Counter()

    nodes_out = []
    edges_out = []
    for entity in entities:
        nodes_out.append({"id": entity, "label": entity, "color": node_color_lookup.get(entity, "#ccc")})

    for subject, predicate, obj in relations:
        edges_out.append({
            "source": subject,
            "target": obj,
            "label": predicate,
        })
        degree[subject] += 1
        degree[obj] += 1
        outdegree[subject] += 1
        indegree[obj] += 1
        predicate_counts[predicate] += 1

    # Also include passage nodes if available
    for pid, pinfo in getattr(graph, 'passages', {}).items():
        nodes_out.append({"id": pid, "label": f"[P] {pid[:8]}...", "color": "#e8d5b7", "type": "passage"})

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "clusters": cluster_view,
        "edge_clusters": edge_cluster_view,
        "stats": {
            "num_nodes": len(nodes_out),
            "num_edges": len(edges_out),
            "num_relations": len(relations),
            "num_passages": len(getattr(graph, 'passages', {})),
            "top_predicates": predicate_counts.most_common(10),
            "top_entities_by_degree": degree.most_common(10),
        },
    }
