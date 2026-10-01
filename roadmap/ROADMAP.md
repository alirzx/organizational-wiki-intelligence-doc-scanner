I want to upgrade the current Paragraph Grouping stage in the `ai-doc-scanner` project, but the goal is not simply to merge adjacent lines.

## Main Goal

The main objective is to preserve **content integrity** during document extraction.

Related, similar, or structurally connected content should not be fragmented into multiple unrelated text blocks.

For example, if the document contains:

```text
Security Requirements:
- User access must be controlled.
- User activities must be logged.
- Logs must be retained.
```

The system should not output every line as an independent unrelated paragraph.

Instead, it should understand that these items belong to one logical content group:

```json
{
  "group_type": "list_section",
  "title": "Security Requirements",
  "members": [
    "User access must be controlled.",
    "User activities must be logged.",
    "Logs must be retained."
  ],
  "confidence": 0.94
}
```

---

# Architectural Constraint

Do not build an LLM-based solution.

The preferred approach is lightweight/classical Machine Learning.

Recommended baseline:

- LightGBM or XGBoost for classification
- Graph-based grouping for final content groups
- Embeddings only as features
- Use `embeddinggemma` through Ollama for semantic embeddings

Ollama is available at:

```text
http://127.0.0.1:11434
```

Embedding model:

```text
embeddinggemma
```

Use the standard Ollama embedding API.

The embedding implementation must be isolated behind an adapter so the provider can easily be replaced later.

Suggested structure:

```text
app/text_processing/embeddings/
    base.py
    ollama_embedding.py
```

Suggested interface:

```python
class TextEmbedder:
    def embed(self, text: str) -> list[float]:
        ...
```

Implementation:

```python
class OllamaEmbeddingGemma(TextEmbedder):
    ...
```

---

# Target Architecture

The new pipeline should be close to:

```text
OCR / Layout
      ↓
Atomic Text Blocks
      ↓
Text Normalization
      ↓
Feature Extraction
      ├── Visual Features
      ├── Structural Features
      ├── Linguistic Features
      └── Semantic Features
               ↓
        embeddinggemma
               ↓
Candidate Neighbor Generation
      ↓
Pairwise ML Classifier
      ↓
Relationship Graph
      ↓
Content Group Resolver
      ↓
Unified Content Blocks
```

The core concept is:

```text
Visual Evidence
+
Structural Similarity
+
Textual Continuity
+
Semantic Similarity
+
ML Classification
=
Content Integrity
```

---

# 1. Atomic Text Blocks

Treat each detected OCR/layout line or text region as an atomic unit before grouping.

Suggested representation:

```python
@dataclass
class TextBlock:
    id: str
    text: str
    bbox: tuple[int, int, int, int]
    page: int

    confidence: float | None = None
    block_type: str | None = None
    column_id: int | None = None

    metadata: dict = field(default_factory=dict)
```

If the project already has an equivalent schema, reuse and extend it instead of introducing a duplicate model.

---

# 2. Text Normalization

Add a text preprocessing layer before feature extraction.

For Persian text, normalize at least:

- `ي` → `ی`
- `ك` → `ک`
- excessive whitespace
- malformed line breaks
- zero-width non-joiner / half-space
- Unicode normalization
- mixed RTL/LTR edge cases
- Persian/English digits where appropriate

Suggested file:

```text
app/text_processing/normalizer.py
```

Keep the original OCR text accessible in metadata.

The normalization behavior should be configurable.

---

# 3. Feature Extraction

For each candidate pair of text blocks, generate a feature vector.

## Visual Features

Include at least:

```text
vertical_gap
horizontal_gap
horizontal_overlap_ratio
vertical_overlap_ratio
indent_delta
left_alignment_delta
right_alignment_delta
bbox_width_ratio
bbox_height_ratio
same_column
same_page
page_distance
```

Also include page-relative positions where useful.

---

## Structural Features

Include at least:

```text
same_numbering_pattern
same_bullet_pattern
same_prefix_pattern
same_indent
similar_line_height
similar_bbox_width
starts_with_list_marker
starts_with_number
starts_with_bullet
heading_like_a
heading_like_b
```

Support common patterns such as:

```text
1.
2.
3.

1)
2)
3)

الف)
ب)
ج)

-
•
▪
```

The system must understand that structurally similar items may belong to the same group even if their semantic similarity is not especially high.

Example:

```text
Name
Family Name
National ID
Phone Number
```

These values may not have strong semantic similarity, but they clearly belong to the same structural group.

---

# 4. Linguistic Features

Include at least:

```text
a_ends_with_sentence_terminator
a_ends_with_colon
a_ends_with_comma
a_ends_with_semicolon
a_sentence_complete
b_starts_with_connector
b_starts_with_list_marker
token_count_a
token_count_b
length_ratio
```

For Persian, detect continuation connectors such as:

```text
و
یا
که
اما
ولی
همچنین
بنابراین
زیرا
در نتیجه
به منظور
به‌طوری‌که
```

Example:

```text
This policy applies to all employees
and contractors working for the organization.
```

These two lines should strongly favor `SAME_GROUP`.

---

# 5. Semantic Features with embeddinggemma

Generate embeddings for each block using Ollama:

```python
embedding_a = embedder.embed(block_a.text)
embedding_b = embedder.embed(block_b.text)
```

Then derive features such as:

```text
cosine_similarity
```

Embedding similarity must only be one feature among many.

Do not make semantic similarity the primary grouping rule.

For performance:

- cache embeddings
- do not embed the same text repeatedly
- implement at least an in-memory cache
- design the interface so Redis caching could be added later
- use batch embedding if supported by the selected Ollama endpoint

---

# 6. Ollama Embedding Adapter

Configure through environment variables:

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=embeddinggemma
```

The adapter must include:

- timeout handling
- connection error handling
- model-unavailable handling
- structured logging

If semantic embeddings are optional, support a configuration such as:

```text
SEMANTIC_FEATURES_ENABLED=true
```

If Ollama becomes unavailable, do not fail with an unclear exception.

Either:

- use a clear exception such as `SemanticEmbeddingUnavailable`, or
- fall back to non-semantic features when configured to do so

The entire document extraction pipeline should not become unusable only because the embedding service is unavailable.

---

# 7. Candidate Neighbor Generation

Do not compare every text block against every other text block.

Create a candidate generator.

Suggested file:

```text
app/text_processing/candidate_generator.py
```

Candidate pairs should primarily include:

- adjacent blocks in reading order
- several nearby blocks after the current block
- blocks in the same column
- spatially nearby blocks
- last block of page N vs first block of page N+1
- structurally similar nearby blocks

This prevents O(N²) pair generation on large documents.

---

# 8. Pairwise ML Classifier

Create a dedicated classifier abstraction.

Prefer LightGBM as the first implementation.

If LightGBM is not appropriate for the current environment, XGBoost is acceptable.

Suggested interface:

```python
class BlockRelationshipClassifier:
    def predict(
        self,
        block_a,
        block_b,
        features
    ) -> RelationshipPrediction:
        ...
```

Suggested output:

```python
@dataclass
class RelationshipPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
```

Start with:

```text
SAME_GROUP
NEW_GROUP
```

But design the abstraction so it can later support:

```text
SAME_GROUP
CONTINUATION
LIST_CHILD
HEADING_CHILD
SEPARATE
```

---

# 9. Hybrid Rules + ML

Do not delete the current heuristic rules.

Convert useful existing heuristics into:

- ML features
- pre-classification signals
- validation guards

The intended design is:

```text
Existing Rules
      ↓
Feature Extraction
      ↓
ML Prediction
      ↓
Validation Guards
```

Example:

A heading and the paragraph immediately below it may have very high semantic similarity.

That does not mean they should be flattened into one plain paragraph.

The system should preserve their structural relationship.

---

# 10. Relationship Graph

Do not immediately concatenate strings after pairwise classification.

Build a relationship graph.

Each `TextBlock` becomes a node.

Predicted relationships become edges.

Example:

```text
A -- SAME_GROUP --> B
B -- SAME_GROUP --> C
C -- NEW_GROUP --> D
```

The final grouping becomes:

```text
Group 1 = A, B, C
Group 2 = D
```

Suggested module:

```text
app/text_processing/content_graph.py
```

A heavy graph library is not required.

A lightweight adjacency structure / connected component resolver is sufficient for the first implementation.

---

# 11. Content Group Resolver

Add a final group resolution stage.

Suggested representation:

```python
@dataclass
class ContentGroup:
    id: str
    group_type: str
    members: list[TextBlock]
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float
    metadata: dict
```

The group bounding box must be the union of all member bounding boxes.

The final text should preserve logical ordering.

Do not simply concatenate raw OCR strings blindly.

---

# 12. Group Type Detection

Support at least:

```text
paragraph
list
list_section
heading_section
generic_group
```

Example:

```text
Security Requirements:
- Log activities
- Control access
- Retain audit records
```

should preferably become:

```text
list_section
```

instead of four independent paragraphs.

---

# 13. Cross-page Integrity

Page boundaries must not automatically break content groups.

Compare:

```text
last block of page N
```

with:

```text
first block of page N+1
```

Include features such as:

```text
cross_page
sentence_incomplete
same_indent
same_structural_pattern
semantic_similarity
same_column_position
```

Example:

Page 5:

```text
This process applies to all employees and
```

Page 6:

```text
contractors working for the organization.
```

These should be recoverable as the same logical paragraph/group.

---

# 14. Training Dataset Format

Define a clean training dataset format.

JSONL is preferred.

Example:

```json
{
  "block_a": {
    "text": "...",
    "bbox": [100, 200, 500, 240],
    "page": 1
  },
  "block_b": {
    "text": "...",
    "bbox": [100, 250, 510, 290],
    "page": 1
  },
  "label": "SAME_GROUP"
}
```

Add a dataset preparation script:

```text
scripts/build_grouping_dataset.py
```

The implementation should make manual annotation reasonably easy.

---

# 15. Training Script

Create:

```text
scripts/train_grouping_model.py
```

The script must:

- load the dataset
- generate features
- split train/validation data
- train LightGBM
- evaluate the model
- save the model artifact
- save feature metadata
- report metrics

At minimum report:

```text
Precision
Recall
F1-score
Confusion Matrix
```

Do not rely on accuracy alone.

If applicable, also include ROC-AUC.

---

# 16. Feature Importance

Generate feature importance reports.

At minimum:

```text
LightGBM feature importance
```

Optionally support SHAP.

The objective is to understand whether the model is relying more on:

```text
vertical_gap
semantic_similarity
indent_delta
list_pattern
sentence_continuity
alignment
...
```

This is important for debugging and model trust.

---

# 17. Confidence Thresholds

Make relationship thresholds configurable.

Example:

```text
GROUPING_MERGE_THRESHOLD=0.75
```

Possible behavior:

```text
confidence >= 0.75
→ SAME_GROUP
```

Optionally introduce an uncertain range:

```text
0.55 - 0.75
→ UNCERTAIN
```

Store uncertain predictions in metadata for later analysis.

---

# 18. Fallback Behavior

The existing pipeline must continue working if the ML model is unavailable.

Required behavior:

```text
ML model available
      ↓
ML grouping

ML model unavailable
      ↓
existing heuristic grouping
```

Do not remove the current grouper until the new ML pipeline is proven better.

This upgrade must be incremental and backward compatible.

---

# 19. Backward Compatibility

Avoid breaking the current API and artifact contract.

Do not remove existing output fields.

Add new fields when needed, for example:

```json
{
  "group_id": "...",
  "group_type": "list_section",
  "group_confidence": 0.91
}
```

Preserve the existing canonical coordinate space.

---

# 20. Testing

Add unit and integration tests.

Important cases:

## Persian Paragraph Continuation

```text
این سند برای کلیه کارکنان
و پیمانکاران سازمان لازم‌الاجرا است.
```

Expected:

```text
SAME_GROUP
```

---

## List Grouping

```text
موارد زیر الزامی است:
- ثبت لاگ
- کنترل دسترسی
- نگهداری سوابق
```

Expected:

One logical content group.

---

## Similar Structured Items

```text
نام
نام خانوادگی
کد ملی
شماره تماس
```

These should be recognized as a structurally related group even if semantic similarity is relatively weak.

---

## Heading + Body

```text
3. دامنه کاربرد

این دستورالعمل برای کلیه واحدهای سازمان اعمال می‌شود.
```

The heading and body should remain structurally connected but should not be flattened into one plain text string.

---

## Visually Close but Unrelated

Two nearby blocks belonging to different sections must not be merged only because their vertical distance is small.

---

## Cross-page Continuation

A paragraph interrupted by a page break should be recoverable as one logical group.

---

# 21. Performance Requirements

Embedding must not become the pipeline bottleneck.

Add timing measurements for:

```text
embedding_time
feature_extraction_time
candidate_generation_time
classifier_time
group_resolution_time
total_grouping_time
```

Also measure:

```text
candidate_pairs_per_document
embedding_cache_hit_rate
```

Avoid unnecessary embedding requests.

---

# Suggested Project Structure

Preserve the current architecture where possible.

A reasonable structure would be:

```text
app/
  text_processing/
    normalizer.py
    linguistic_features.py
    structural_features.py
    visual_features.py
    feature_extractor.py

    embeddings/
      base.py
      ollama_embedding.py

    candidate_generator.py

    classifiers/
      base.py
      lightgbm_classifier.py

    relationship.py
    content_graph.py
    group_resolver.py

  orchestration/
    paragraph_resolver.py
```

Do not create duplicate abstractions if equivalent modules already exist in the project.

Integrate with the existing architecture.

---

# Critical Design Principle

The goal is not:

```text
merge lines that are close together
```

The goal is:

```text
Content Integrity Preservation
```

The system should preserve logical units such as:

```text
paragraphs
lists
list sections
heading + body relationships
repeated structured fields
cross-page content
```

The final decision must combine:

```text
Visual Evidence
+
Structural Evidence
+
Linguistic Evidence
+
Semantic Evidence
+
Machine Learning
```

`embeddinggemma` should provide semantic evidence, but it must not become the sole decision-maker.

---

# Before Modifying the Code

First inspect the repository and identify:

1. where current paragraph grouping happens
2. the current text-block schema
3. existing preprocessing stages
4. where reading order is calculated
5. existing paragraph/grouping tests
6. current artifact/output contracts
7. the safest integration points

Then implement the change.

Do not rewrite the project from scratch.

Do not replace working architecture unnecessarily.

The implementation must be incremental.

---

# Final Deliverables

After implementation, provide:

1. list of changed files
2. explanation of the new architecture
3. Ollama + `embeddinggemma` setup instructions
4. new environment variables
5. dataset format
6. dataset-building instructions
7. classifier training instructions
8. model artifact location
9. ML grouping enable/disable configuration
10. fallback behavior
11. tests added
12. before/after examples
13. basic latency benchmark
14. feature importance report
15. remaining limitations

Also update the README with a new section:

```text
ML-based Content Integrity Grouping
```

Do not stop after creating only abstractions or placeholder code.

Implement the actual working pipeline and connect it to the existing extraction flow.