"""Judge prompt templates. Bump PROMPT_VERSION on any edit (cache key input)."""

from __future__ import annotations

PROMPT_VERSION = "v3.0-identity"

# Calibration examples, balanced and drawn from REAL failures observed on the
# corpus (apple+pear, fish+marine fish, cow-milk fat variants — the judge
# wrongly merged these on "same category / co-occur in text"). Toggle via
# `cfg.use_few_shot`.
FEW_SHOT_EXAMPLES = """\
EXAMPLES (calibration only — not part of the task):

MERGE (the labels name the SAME food; differ only in spelling/synonym/form):
- "yoghurt" vs "yogurt food product" -> merge. Spelling / labelling variant.
- "chickpea" vs "garbanzo bean" -> merge. Two common names for one food.
- "ground beef" vs "minced beef" -> merge. Processing-synonym for one food.

KEEP SEPARATE (these are DIFFERENT foods or a food vs its category — never
merge them, even though they co-occur in text or share a category):
- "apple" vs "pear" -> keep separate. Different fruits. Co-occurring in
  "healthy eating" text does NOT make them the same food.
- "pea" vs "tomato" -> keep separate. Different vegetables.
- "fish" vs "marine fish" -> keep separate. A category and its subtype.
- "cow milk" vs "cow whole milk" -> keep separate. Fat level is a meaningful
  distinction; do not collapse into a broader bucket.
- "olive oil" vs "vegetable oil" -> keep separate. Specific food vs category.
"""

# The cluster is rendered as numbered shelf blocks; the model answers by index
# (1-based) so it never has to echo opaque shelf ids.
JUDGE_CLUSTER_PROMPT = """\
You are a domain expert deduplicating shelves in a nutrition knowledge graph.

An embedding model grouped the {n} shelves below because their text is similar.
Similar text is NOT enough to merge. Apply ONE strict test:

  Merge two shelves ONLY IF their labels name the exact SAME food/concept,
  differing only in spelling, singular/plural, a synonym, or a
  processing/labelling variant (e.g. "yoghurt" = "yogurt", "chickpea" =
  "garbanzo bean").

Do NOT merge when, even if they look or read similarly:
  - they are DIFFERENT foods (apple vs pear, pea vs tomato) — being in the same
    category or co-occurring in the same chunks is NOT a reason to merge;
  - one is a CATEGORY and the other a member of it (fish vs marine fish, oil vs
    olive oil, milk vs whole milk);
  - they differ by a nutritionally meaningful attribute (fat level, raw vs
    cooked, sweetened vs unsweetened).

When unsure, KEEP SEPARATE. Use the sample chunks only to confirm that two
identical-looking labels really mean the same food — never to merge two
different foods because they appear together.
{few_shot}
Shelves:
{shelf_blocks}

Respond with a JSON object:
{{"merge_groups": [{{"members": [<1-based indices naming the SAME food>],
                     "canonical_name": "<the food's name>",
                     "confidence": <0.0-1.0>,
                     "rationale": "<why these are the same food>"}}],
  "keep_alone": [<1-based indices that stay distinct>]}}

Every index 1..{n} must appear exactly once, in one merge_groups entry (groups
have 2+ members) or in keep_alone. Output JSON only.
"""

JUDGE_CLUSTER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "merge_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "members": {"type": "array", "items": {"type": "integer"}},
                    "canonical_name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["members", "canonical_name", "confidence", "rationale"],
            },
        },
        "keep_alone": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["merge_groups", "keep_alone"],
}
