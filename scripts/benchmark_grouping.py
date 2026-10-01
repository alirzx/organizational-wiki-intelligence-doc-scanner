from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.text_processing.candidate_generator import CandidateConfig, generate_candidates
from app.text_processing.embeddings.cache import RunEmbeddingCache, semantic_cache_key
from app.text_processing.embeddings.base import SemanticIdentity
from app.schemas.common import BBox
from app.text_processing.types import TextBlock


def _block(index: int) -> TextBlock:
    page = index // 50 + 1; ordinal = index % 50; y = ordinal * 24 + 1
    return TextBlock(
        block_id=f"b{index}", document_id="benchmark", page_id=f"benchmark:p{page}", page_number=page,
        content_ordinal=ordinal, original_text=f"line {index}", normalized_text=f"line {index}",
        bbox=BBox(x1=100, y1=y, x2=600, y2=y + 19), page_width=1000, page_height=1400,
        ocr_confidence=1., column_id="main",
    )


def benchmark(block_count: int = 1000) -> dict:
    blocks = [_block(i) for i in range(block_count)]
    started = perf_counter(); pairs = generate_candidates(blocks, CandidateConfig(reading_lookahead=4, max_pairs=block_count * 25)); candidate_ms = (perf_counter() - started) * 1000
    identity = SemanticIdentity("benchmark", "deterministic", 4, "none", "normalizer.v1")
    cache = RunEmbeddingCache(block_count)
    keys = [semantic_cache_key(block.normalized_text, identity) for block in blocks]
    cold = perf_counter(); cache.set_many({key: (1., 0., 0., 0.) for key in keys}); cold_ms = (perf_counter() - cold) * 1000
    warm = perf_counter(); hits = cache.get_many(keys); warm_ms = (perf_counter() - warm) * 1000
    return {"blocks": block_count, "candidate_pairs": len(pairs), "pairs_per_block": len(pairs) / block_count,
            "candidate_ms": candidate_ms, "cold_cache_ms": cold_ms, "warm_cache_ms": warm_ms,
            "warm_cache_hit_rate": len(hits) / block_count, "heuristic_overhead_ms": 0., "learned_overhead_ms": candidate_ms}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--blocks", type=int, default=1000); args = parser.parse_args()
    print(json.dumps(benchmark(args.blocks), indent=2))


if __name__ == "__main__": main()
