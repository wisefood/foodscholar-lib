"""Pydantic data carrier for ontology terms.

Lives in `io/` alongside the other contracts so phase modules import a stable
shape regardless of which ontology backend produced the data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

OntologyId = str


class OntologyTerm(BaseModel):
    """A flattened, query-ready ontology term.

    `ancestor_ids` is the *closed transitive* set of ancestors so downstream
    phases (layer_a propagation, the linker's semantic-type gate) don't have
    to re-walk the tree. `parent_ids` is the direct-parent set, kept separate
    for tree walks.
    """

    model_config = ConfigDict(frozen=True)

    id: OntologyId
    label: str
    synonyms: tuple[str, ...] = Field(default_factory=tuple)
    related_synonyms: tuple[str, ...] = Field(default_factory=tuple)
    parent_ids: tuple[OntologyId, ...] = Field(default_factory=tuple)
    ancestor_ids: tuple[OntologyId, ...] = Field(default_factory=tuple)
    obsolete: bool = False
