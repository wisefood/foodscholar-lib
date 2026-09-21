"""Extended KGGen orchestrator with passage-aware knowledge graph extraction.

Extends the original KGGen to track which source passage each triplet was
extracted from. Key additions:
  - generate() accepts passage_id, passage_text, passage_metadata
  - Records entity_passages and triplet_passages during extraction
  - aggregate() merges passage-aware fields across graphs
  - deduplicate() returns ExtendedGraph with remapped passage tracking
"""

from typing import Union, List, Dict, Optional, Any, Set, Tuple
from typing_extensions import deprecated

from kggen_extended.steps._1_get_entities import get_entities
from kggen_extended.steps._2_get_relations import get_relations
from kggen_extended.steps._3_deduplicate import (
    run_deduplication,
    DeduplicateMethod,
)
from kggen_extended.utils.chunk_text import chunk_text
from kggen_extended.utils.visualize_kg import visualize_kg as _visualize_kg
from kggen_extended.models import ExtendedGraph, _triple_to_key, _key_to_triple
import dspy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure dspy logging to only show errors
import logging

logger = logging.getLogger(__name__)

dspy_logger = logging.getLogger("dspy")
dspy_logger.setLevel(logging.CRITICAL)


class ExtendedKGGen:
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_tokens: int = 16000,
        temperature: float = 0.0,
        reasoning_effort: str = None,
        api_key: str = None,
        api_base: str = None,
        retrieval_model: Optional[str] = None,
        disable_cache: bool = False,
    ):
        """Initialize ExtendedKGGen with optional model configuration

        Args:
            model: Name of model to use (e.g. 'gpt-4')
            temperature: Temperature for model sampling
            api_key: API key for model access
            api_base: Specify the base URL endpoint for making API calls to a language model service
        """
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.retrieval_model: Optional[SentenceTransformer] = None
        self.lm = None
        self.disable_cache = disable_cache

        self.init_model(
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            retrieval_model=retrieval_model,
        )

    def validate_temperature(self, temperature: float):
        if "gpt-5" in self.model and temperature < 1.0:
            raise ValueError("Temperature must be 1.0 for gpt-5 family models")

    def validate_max_tokens(self, max_tokens: int):
        if "gpt-5" in self.model and max_tokens < 16000:
            raise ValueError("Max tokens must be 16000 for gpt-5 family models")

    def init_model(
        self,
        model: str = None,
        reasoning_effort: str = None,
        max_tokens: int = None,
        temperature: float = None,
        retrieval_model: str = None,
        api_key: str = None,
        api_base: str = None,
    ):
        """Initialize or reinitialize the model with new parameters"""
        # Update instance variables if new values provided
        if model is not None:
            self.model = model
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if api_key is not None:
            self.api_key = api_key
        if api_base is not None:
            self.api_base = api_base
        if temperature is not None:
            self.temperature = temperature
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
        if retrieval_model is not None:
            self.retrieval_model = SentenceTransformer(retrieval_model)

        self.validate_temperature(self.temperature)
        self.validate_max_tokens(self.max_tokens)

        # Initialize dspy LM with current settings
        if self.api_key:
            self.lm = dspy.LM(
                model=self.model,
                api_key=self.api_key,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                cache=not self.disable_cache,
                model_type="responses" if self.model.startswith("openai/") else "chat",
            )
        else:
            self.lm = dspy.LM(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_base=self.api_base,
                reasoning={"effort": self.reasoning_effort}
                if self.reasoning_effort
                else None,
                cache=not self.disable_cache,
                model_type="responses" if self.model.startswith("openai/") else "chat",
            )

    @staticmethod
    def from_file(file_path: str) -> ExtendedGraph:
        return ExtendedGraph.from_file(file_path)

    @staticmethod
    def from_dict(graph_dict: dict) -> ExtendedGraph:
        return ExtendedGraph(**graph_dict)

    # ── EXTENDED: generate with passage tracking ──────────────────

    def generate(
        self,
        input_data: Union[str, List[Dict]],
        model: str = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
        chunk_size: Optional[int] = None,
        reasoning_effort: str = None,
        deduplication_method: DeduplicateMethod | None = DeduplicateMethod.SEMHASH,
        temperature: float = None,
        output_folder: Optional[str] = None,
        no_dspy: bool = False,
        # ── EXTENDED: passage tracking params ──
        passage_id: Optional[str] = None,
        passage_text: Optional[str] = None,
        passage_metadata: Optional[dict] = None,
    ) -> ExtendedGraph:
        """Generate a knowledge graph from input text with passage tracking.

        Args:
            input_data: Text string or list of message dicts
            model: Name of OpenAI model to use
            api_key (str): OpenAI API key for making model calls
            chunk_size: Max size of text chunks in characters to process
            context: Description of data context
            output_folder: Path to save partial progress
            passage_id: UUID of the source passage (for provenance tracking)
            passage_text: Original text of the source passage
            passage_metadata: Dict with metadata about the passage (file, page, etc.)

        Returns:
            ExtendedGraph: Generated knowledge graph with passage provenance
        """

        # Process input data
        is_conversation = isinstance(input_data, list)
        if is_conversation:
            text_content = []
            for message in input_data:
                if (
                    not isinstance(message, dict)
                    or "role" not in message
                    or "content" not in message
                ):
                    raise ValueError(
                        "Messages must be dicts with 'role' and 'content' keys"
                    )
                if message["role"] in ["user", "assistant"]:
                    text_content.append(f"{message['role']}: {message['content']}")

            processed_input = "\n".join(text_content)
        else:
            processed_input = input_data

        # Reinitialize dspy with new parameters if any are provided
        if any([model, temperature, api_key, api_base, reasoning_effort]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
                reasoning_effort=reasoning_effort or self.reasoning_effort,
            )

        # ── EXTENDED: Track per-chunk passage provenance ──
        # We track entity_passages and triplet_passages as we extract.
        # If passage_id is provided, all chunks inherit it.
        entity_passages: Dict[str, Set[str]] = {}
        triplet_passages: Dict[str, Set[str]] = {}  # string keys ('s::p::o')

        def _process(content, lm):
            with dspy.context(lm=lm):
                entities = get_entities(
                    content,
                    is_conversation,
                    use_litellm_prompt=no_dspy,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=temperature
                    if temperature is not None
                    else self.temperature,
                )
                relations = get_relations(
                    content,
                    entities,
                    is_conversation=is_conversation,
                    use_litellm_prompt=no_dspy,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=temperature
                    if temperature is not None
                    else self.temperature,
                )
                return entities, relations

        if not chunk_size:
            try:
                entities, relations = _process(processed_input, self.lm)
            except Exception as e:
                if "context length" in str(e).lower():
                    logger.warning(
                        f"Context length error: {e}. Chunking text with chunk size 16384."
                    )
                    chunk_size = 16384
                else:
                    raise e

        if chunk_size:
            chunks = chunk_text(processed_input, chunk_size)
            entities = set()
            relations = set()

            with ThreadPoolExecutor() as executor:
                future_to_chunk = {
                    executor.submit(_process, chunk, self.lm): chunk for chunk in chunks
                }

                for future in as_completed(future_to_chunk):
                    chunk_entities, chunk_relations = future.result()
                    entities.update(chunk_entities)
                    relations.update(chunk_relations)
        else:
            # Convert list to set of tuples
            relations = set(relations)
            entities = set(entities)

        # ── EXTENDED: Record passage provenance ──
        pid = passage_id
        if pid:
            for entity in entities:
                if entity not in entity_passages:
                    entity_passages[entity] = set()
                entity_passages[entity].add(pid)

            for triple in relations:
                tkey = _triple_to_key(triple)
                if tkey not in triplet_passages:
                    triplet_passages[tkey] = set()
                triplet_passages[tkey].add(pid)

        # Build passage registry entry
        passages_registry: dict[str, dict[str, Any]] = {}
        if pid and passage_text is not None:
            passages_registry[pid] = {
                "text": passage_text,
                "metadata": passage_metadata or {},
            }

        graph = ExtendedGraph(
            entities=entities,
            relations=relations,
            edges={relation[1] for relation in relations},
            passages=passages_registry,
            entity_passages=entity_passages if entity_passages else None,
            triplet_passages=triplet_passages if triplet_passages else None,
        )

        if deduplication_method:
            graph = self.deduplicate(
                graph, method=deduplication_method, context=context
            )

        if output_folder:
            graph.to_file(os.path.join(output_folder, "graph.json"))
        return graph

    @deprecated("Use ExtendedKGGen.deduplicate() method instead")
    def cluster(
        self,
        graph: ExtendedGraph,
        **kwargs,
    ) -> ExtendedGraph:
        return self.deduplicate(graph, **kwargs)

    # ── EXTENDED: deduplicate with ExtendedGraph return ───────────

    def deduplicate(
        self,
        graph: ExtendedGraph,
        method: DeduplicateMethod = DeduplicateMethod.FULL,
        semhash_similarity_threshold: float = 0.95,
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
    ) -> ExtendedGraph:
        # Reinitialize dspy with new parameters if any are provided
        if any([model, temperature, api_key, api_base]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
            )

        return run_deduplication(
            lm=self.lm,
            graph=graph,
            method=method,
            retrieval_model=self.retrieval_model,
            semhash_similarity_threshold=semhash_similarity_threshold,
        )

    # ── EXTENDED: aggregate with passage-aware merge ──────────────

    def aggregate(self, graphs: List[ExtendedGraph]) -> ExtendedGraph:
        # Initialize empty sets for combined graph
        all_entities = set()
        all_relations = set()
        all_edges = set()
        all_entity_metadata: Dict[str, Set[str]] = {}

        # ── EXTENDED: Passage-aware aggregation ──
        all_passages: Dict[str, Dict[str, Any]] = {}
        all_entity_passages: Dict[str, Set[str]] = {}
        all_triplet_passages: Dict[str, Set[str]] = {}

        # Combine all graphs
        for graph in graphs:
            all_entities.update(graph.entities)
            all_relations.update(graph.relations)
            all_edges.update(graph.edges)

            if graph.entity_metadata:
                for entity, metadata_set in graph.entity_metadata.items():
                    if entity in all_entity_metadata:
                        all_entity_metadata[entity].update(metadata_set)
                    else:
                        all_entity_metadata[entity] = metadata_set.copy()

            # ── EXTENDED: Merge passage registry ──
            if graph.passages:
                for pid, pinfo in graph.passages.items():
                    if pid not in all_passages:
                        all_passages[pid] = pinfo
                    else:
                        # Merge metadata if passage appears in multiple graphs
                        if "metadata" in pinfo and "metadata" in all_passages[pid]:
                            all_passages[pid]["metadata"].update(pinfo["metadata"])

            # ── EXTENDED: Merge entity_passages ──
            if graph.entity_passages:
                for entity, pids in graph.entity_passages.items():
                    if entity in all_entity_passages:
                        all_entity_passages[entity].update(pids)
                    else:
                        all_entity_passages[entity] = set(pids)

            # ── EXTENDED: Merge triplet_passages (keys are serialized strings) ──
            if graph.triplet_passages:
                for tkey, pids in graph.triplet_passages.items():
                    if tkey in all_triplet_passages:
                        all_triplet_passages[tkey].update(pids)
                    else:
                        all_triplet_passages[tkey] = set(pids)

        # Create and return aggregated ExtendedGraph
        return ExtendedGraph(
            entities=all_entities,
            relations=all_relations,
            edges=all_edges,
            entity_metadata=all_entity_metadata if all_entity_metadata else None,
            passages=all_passages,
            entity_passages=all_entity_passages if all_entity_passages else None,
            triplet_passages=all_triplet_passages if all_triplet_passages else None,
        )

    @staticmethod
    def visualize(graph: ExtendedGraph, output_path: str, open_in_browser: bool = False):
        _visualize_kg(graph, output_path, open_in_browser=open_in_browser)

    # ====== Retrieval Methods ======

    def _parse_embedding_model(
        self, model: Optional[SentenceTransformer] = None
    ) -> Optional[SentenceTransformer]:
        if model is None:
            model = self.retrieval_model
        if model is None:
            raise ValueError("No retrieval model provided")
        return model

    @staticmethod
    def to_nx(graph: ExtendedGraph) -> nx.DiGraph:
        """Convert ExtendedGraph to networkx DiGraph (entity nodes only)."""
        G = nx.DiGraph()
        for entity in graph.entities:
            G.add_node(entity)

        for relation in graph.relations:
            source, rel, target = relation
            G.add_edge(source, target, relation=rel)
        return G

    def generate_embeddings(
        self,
        graph: Union[ExtendedGraph, nx.DiGraph],
        model: Optional[SentenceTransformer] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        model = self._parse_embedding_model(model)
        if isinstance(graph, ExtendedGraph):
            graph = self.to_nx(graph)

        node_embeddings = {node: model.encode(node).tolist() for node in graph.nodes}
        relation_embeddings = {
            rel: model.encode(rel).tolist()
            for rel in set(edge[2]["relation"] for edge in graph.edges(data=True))
        }
        return node_embeddings, relation_embeddings

    def retrieve(
        self,
        query: str,
        node_embeddings: dict[str, np.ndarray],
        graph: nx.DiGraph,
        model: Optional[SentenceTransformer] = None,
        k: int = 8,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[str, float]], Set[str], str]:
        model = self._parse_embedding_model(model)
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)
        return top_nodes, set(), ""

    def retrieve_relevant_nodes(
        self,
        query: str,
        node_embeddings: dict[str, np.ndarray],
        model: Optional[SentenceTransformer] = None,
        k: int = 8,
    ) -> List[Tuple[str, float]]:
        model = self._parse_embedding_model(model)
        query_embedding = model.encode(query)
        similarities = {}
        for node, emb in node_embeddings.items():
            sim = cosine_similarity([query_embedding], [emb])[0][0]
            similarities[node] = float(sim)
        sorted_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:k]
