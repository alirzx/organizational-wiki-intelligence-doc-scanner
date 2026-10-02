"""Measure real grouping stages; synthetic workloads do not establish accuracy."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.orchestration.paragraph_resolver import resolve_paragraphs
from app.schemas.common import BBox
from app.modules.ocr.adapter import groups_to_detected_objects
from app.text_processing.classifiers.clustering_classifier import ClusteringRelationshipClassifier
from app.text_processing.classifiers.lightgbm_classifier import LightGBMRelationshipClassifier
from app.text_processing.candidate_generator import CandidateConfig
from app.text_processing.feature_extractor import feature_schema
from app.text_processing.types import TextBlock
from app.text_processing.canonical import canonical_block
from app.text_processing.normalizer import TextNormalizer
from app.text_processing.embeddings.cache import RunEmbeddingCache
from app.text_processing.embeddings.ollama_embedding import OllamaEmbedder

SCENARIOS = ('single-column','multi-column','list-heavy','heading-heavy','cross-page','persian','english','mixed')


def _block(index: int) -> TextBlock:
    page = index // 50 + 1; ordinal = index % 50; y = ordinal * 24 + 1
    return TextBlock(block_id=f'b{index}',document_id='benchmark',page_id=f'benchmark:p{page}',
        page_number=page,content_ordinal=ordinal,original_text=f'line {index}',normalized_text=f'line {index}',
        bbox=BBox(x1=100,y1=y,x2=600,y2=y+19),page_width=1000,page_height=1400,ocr_confidence=1.,column_id='main')


def scenario_blocks(n, scenario):
    if scenario not in SCENARIOS:
        raise ValueError('unsupported benchmark scenario')
    blocks = [_block(i) for i in range(n)]
    result = []
    for i,b in enumerate(blocks):
        text = b.original_text
        if scenario == 'persian' or scenario == 'mixed' and i%2:
            text = f'این متن ادامه یک پاراگراف فارسی است {i}'
        elif scenario == 'english':
            text = f'this is a continued English paragraph line {i}'
        elif scenario == 'list-heavy':
            text = f'{i+1}. List item' if i%3 == 0 else 'and continued list detail'
        if scenario == 'heading-heavy':
            b = replace(b,block_type='heading' if i%10 == 0 else 'body')
            text = 'Section title' if i%10 == 0 else 'and body text continues'
        if scenario == 'multi-column':
            right = b.content_ordinal >= 25
            y = (b.content_ordinal%25)*24+1
            b = replace(b,column_id='right' if right else 'left',
                        bbox=BBox(x1=650 if right else 50,y1=y,x2=950 if right else 350,y2=y+19))
        if scenario == 'cross-page':
            text = 'and paragraph continues across the page'
        result.append(replace(b,original_text=text,normalized_text=text))
    return result


def _embedder(semantics, n):
    if semantics == 'off':
        return None
    if semantics == 'mock':
        import httpx
        def handler(request):
            payload = json.loads(request.content)
            return httpx.Response(200,json={'embeddings':[[1.]+[0.]*127 for _ in payload['input']]})
        return OllamaEmbedder(base_url='http://benchmark',dimensions=128,cache=RunEmbeddingCache(max(n,1)),
                              transport=httpx.MockTransport(handler))
    from app.core.runtime import build_grouping_embedder
    from app.core.config import Settings
    return build_grouping_embedder(Settings(semantic_features_enabled=True))


def benchmark(block_count: int = 1000, *, backend='clustering', model_package=None,
              scenario='single-column', semantics='off', warm_cache=False, repeats=3,
              trace_memory=False, cold_model=False) -> dict:
    if block_count < 1 or repeats < 1:
        raise ValueError('positive block count and repetitions required')
    if backend not in {'clustering','lightgbm'} or semantics not in {'off','mock','ollama'}:
        raise ValueError('unsupported benchmark mode')
    if backend == 'lightgbm' and model_package is None:
        raise ValueError('LightGBM benchmark requires a real native package')
    def classifier_factory():
        return (ClusteringRelationshipClassifier() if backend == 'clustering'
                else LightGBMRelationshipClassifier(model_package,feature_schema()))
    classifier = classifier_factory()
    if backend == 'lightgbm' and not cold_model:
        classifier._load()
    config = CandidateConfig(reading_lookahead=4,max_pairs=block_count*25)
    blocks = scenario_blocks(block_count,scenario)
    # Warm optional imports and runtime code outside measured samples.
    resolve_paragraphs(blocks[:min(20,block_count)],classifier,candidate_config=config)
    embedder = _embedder(semantics,block_count)
    samples = []
    try:
        if warm_cache and embedder:
            resolve_paragraphs(blocks,classifier,candidate_config=config,embedder=embedder,semantic_failure_policy='fail_fast')
        for index in range(repeats):
            if embedder and not warm_cache:
                embedder.cache = RunEmbeddingCache(max(block_count,1))
            started = perf_counter()
            normalizer = TextNormalizer()
            normalized = [normalizer.normalize(b.original_text) for b in blocks]
            normalization_ms = (perf_counter()-started)*1000
            block_started = perf_counter()
            canonical = [canonical_block(document_id=b.document_id,page_id=b.page_id,page_number=b.page_number,
                ordinal=b.content_ordinal,text=b.original_text,bbox=b.bbox,page_width=b.page_width,
                page_height=b.page_height,confidence=b.ocr_confidence,column_id=b.column_id,
                block_type=b.block_type,normalization=text) for b,text in zip(blocks,normalized,strict=True)]
            block_ms = (perf_counter()-block_started)*1000
            setup_started = perf_counter()
            active = classifier_factory() if cold_model else classifier
            setup_ms = (perf_counter()-setup_started)*1000
            result = resolve_paragraphs(canonical,active,candidate_config=config,embedder=embedder,semantic_failure_policy='fail_fast')
            projection_started = perf_counter()
            objects = groups_to_detected_objects(result.groups,document_id='benchmark',backend_name='grouping',model_id=active.package_id)
            projection_ms = (perf_counter()-projection_started)*1000
            samples.append({**result.metrics,'normalization_time':normalization_ms,'block_time':block_ms,
                            'model_setup_time':setup_ms,'projection_time':projection_ms,
                            'pipeline_total_time':(perf_counter()-started)*1000})
        median = {key:statistics.median(sample[key] for sample in samples) for key in samples[0]}
        output = {'blocks':block_count,'backend':backend,'scenario':scenario,'semantics':semantics,
                  'cache':'warm' if warm_cache else 'cold','model':'cold' if cold_model else 'warm',
                  'repeats':repeats,'timings_ms':median,'candidate_pairs':len(result.predictions),
                  'candidate_diagnostics':result.candidate_diagnostics,'embedding':result.embedding_diagnostics,
                  'groups':len(result.groups),'projected_objects':len(objects),'samples_ms':samples,
                  'platform':platform.platform(),'python':platform.python_version()}
        if repeats >= 20:
            output['p95_resolver_ms'] = sorted(sample['total_time'] for sample in samples)[__import__('math').ceil(.95*repeats)-1]
        if trace_memory:
            tracemalloc.start()
            resolve_paragraphs(canonical,classifier,candidate_config=config,embedder=embedder,semantic_failure_policy='fail_fast')
            output['peak_traced_python_mib'] = tracemalloc.get_traced_memory()[1]/1024**2
            tracemalloc.stop()
        return output
    finally:
        if embedder:
            embedder.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks',type=int,default=1000); parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--backend',choices=['clustering','lightgbm'],default='clustering')
    parser.add_argument('--model-package',type=Path); parser.add_argument('--scenario',choices=SCENARIOS,default='single-column')
    parser.add_argument('--semantics',choices=['off','mock','ollama'],default='off')
    parser.add_argument('--warm-cache',action='store_true'); parser.add_argument('--trace-memory',action='store_true')
    parser.add_argument('--cold-model',action='store_true'); parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = benchmark(args.blocks,backend=args.backend,model_package=args.model_package,scenario=args.scenario,
                       semantics=args.semantics,warm_cache=args.warm_cache,repeats=args.repeats,
                       trace_memory=args.trace_memory,cold_model=args.cold_model)
    text = json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
