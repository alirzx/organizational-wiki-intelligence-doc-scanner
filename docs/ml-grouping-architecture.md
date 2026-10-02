# OCR grouping architecture

Before: OCR lines → irreversible geometric paragraphs → UUID paragraph objects → ad hoc TextBlocks → all-page candidates → separate training/runtime features → pair classifier → quadratic endpoint reconstruction → groups.

After: retained OCR lines → canonical source-coordinate TextBlocks → shared normalization and stable IDs → bounded local candidates → optional embeddings for participating blocks → shared pair features → batched ML classifier → guarded sparse union-find → content groups → existing page paragraph projection.

`OCRAnalysis` remains internal. Geometric paragraph objects remain available for grouping-disabled responses and error fallback. Providers without raw lines use the canonical object adapter; this cannot recover line information that the provider did not supply.

The feature contract is versioned independently of the public API. Models trained under the previous divergent feature implementation must be retrained. Runtime threshold precedence is explicit configuration, package metadata, then application defaults. Classifier and provider work executes outside the API event loop, with document-local clustering state and bounded process-local infrastructure.
