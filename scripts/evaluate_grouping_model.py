from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import Counter, defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def evaluate_scenarios(rows: list[dict]) -> dict:
    required = {'predicted_group','expected_group','predicted_type','expected_type'}
    if not rows or any(not isinstance(row,dict) or not required <= row.keys()
                       or any(not isinstance(row[key],(str,int)) or row[key] == '' for key in required) for row in rows):
        raise ValueError('evaluation requires nonempty rows with expected and predicted groups/types')
    scenarios: dict[str, list[dict]] = {}
    for row in rows: scenarios.setdefault(row.get("scenario", "unspecified"), []).append(row)
    details = {}
    for name, items in sorted(scenarios.items()):
        predicted_members,expected_members = defaultdict(set),defaultdict(set)
        for index,item in enumerate(items):
            predicted_members[item['predicted_group']].add(index)
            expected_members[item['expected_group']].add(index)
        correct_membership = sum(predicted_members[item['predicted_group']] == expected_members[item['expected_group']] for item in items)
        correct_type = sum(item.get("predicted_type") == item.get("expected_type") for item in items)
        incorrect_merges = sum(bool(item.get("incorrect_merge")) for item in items)
        fragmentation = sum(bool(item.get("fragmented")) for item in items)
        details[name] = {"membership_accuracy": correct_membership / len(items), "type_accuracy": correct_type / len(items),
                         "incorrect_merges": incorrect_merges, "fragmentation": fragmentation,
                         "passed": incorrect_merges == 0 and fragmentation == 0 and correct_membership == len(items) and correct_type == len(items)}
    return {"scenarios": details, "pass_rate": sum(v["passed"] for v in details.values()) / max(1, len(details))}


def evaluate_documents(documents: list[dict], classifier_factory, *, embedder=None) -> dict:
    """Execute the actual resolver against labeled canonical document blocks."""
    from app.text_processing.canonical import block_from_record
    from app.text_processing.candidate_generator import CandidateConfig, generate_candidates
    from app.orchestration.paragraph_resolver import resolve_paragraphs
    from app.text_processing.types import GroupingMode, RelationshipDecision
    from sklearn.metrics import adjusted_rand_score
    if not documents:
        raise ValueError('held-out documents must be nonempty')
    results = []
    for document in documents:
        if not all(key in document for key in ('document_id','blocks','expected_groups')) or not document['blocks']:
            raise ValueError('document requires blocks and expected groups')
        blocks = [block_from_record(row,document['document_id']) for row in document['blocks']]
        source_ids = [row['block_id'] for row in document['blocks']]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError('evaluation source block IDs must be unique')
        mapping = dict(zip(source_ids,[b.block_id for b in blocks],strict=True))
        expected,expected_types = {},{}
        for index,group in enumerate(document['expected_groups']):
            if not group.get('member_ids') or not isinstance(group.get('group_type'),str):
                raise ValueError('expected group requires member IDs and type')
            for source_id in group['member_ids']:
                if source_id not in mapping or mapping[source_id] in expected:
                    raise ValueError('expected groups must partition source blocks')
                expected[mapping[source_id]] = index
                expected_types[mapping[source_id]] = group['group_type']
        if set(expected) != {b.block_id for b in blocks}:
            raise ValueError('expected groups must partition every block')
        classifier = classifier_factory()
        resolution = resolve_paragraphs(blocks,classifier,embedder=embedder)
        config = CandidateConfig(adjacent_only=classifier.grouping_mode == GroupingMode.CLUSTERED)
        candidates = {p.pair_id:p for p in generate_candidates(blocks,config)}
        predicted,predicted_types = {},{}
        for index,group in enumerate(resolution.groups):
            for member in group.members:
                predicted[member.block_id] = index
                predicted_types[member.block_id] = group.group_type.value
        contingency = Counter((expected[b.block_id],predicted[b.block_id]) for b in blocks)
        expected_sizes,predicted_sizes = Counter(expected.values()),Counter(predicted.values())
        n = len(blocks)
        precision = sum(count*count/predicted_sizes[p] for (e,p),count in contingency.items())/n
        recall = sum(count*count/expected_sizes[e] for (e,p),count in contingency.items())/n
        positive_total = sum(size*(size-1)//2 for size in expected_sizes.values())
        candidate_positive = sum(expected[c.block_a_id] == expected[c.block_b_id] for c in candidates.values())
        tp=fp=fn=0
        for prediction in resolution.predictions:
            candidate = candidates[prediction.pair_id]
            actual = expected[candidate.block_a_id] == expected[candidate.block_b_id]
            decision = prediction.decision == RelationshipDecision.MERGE
            tp += actual and decision; fp += not actual and decision; fn += actual and not decision
        pair_precision = tp/(tp+fp) if tp+fp else 0.
        pair_recall = tp/(tp+fn) if tp+fn else 0.
        joint_pairs = sum(size*(size-1)//2 for size in contingency.values())
        predicted_pairs = sum(size*(size-1)//2 for size in predicted_sizes.values())
        ordered = sorted(blocks,key=lambda b:b.document_order)
        adjacent = [(a,b) for a,b in zip(ordered,ordered[1:]) if a.page_number == b.page_number and expected[a.block_id] == expected[b.block_id]]
        boundary = [(a,b) for a,b in zip(ordered,ordered[1:]) if b.page_number == a.page_number+1 and expected[a.block_id] == expected[b.block_id]]
        edges = {(c.block_a_id,c.block_b_id) for c in candidates.values()}
        fragmented = sum(len({p for (e,p) in contingency if e == expected_id})>1 for expected_id in expected_sizes)
        overmerged = sum(len({e for (e,p) in contingency if p == predicted_id})>1 for predicted_id in predicted_sizes)
        results.append({'document_id':document['document_id'],
            'candidate_recall':candidate_positive/positive_total if positive_total else None,
            'adjacent_edge_recall':sum((a.block_id,b.block_id) in edges for a,b in adjacent)/len(adjacent) if adjacent else None,
            'boundary_edge_recall':sum((a.block_id,b.block_id) in edges for a,b in boundary)/len(boundary) if boundary else None,
            'pair_precision':pair_precision,'pair_recall':pair_recall,
            'pair_f1':2*pair_precision*pair_recall/(pair_precision+pair_recall) if pair_precision+pair_recall else 0.,
            'incorrect_merge_rate':(predicted_pairs-joint_pairs)/predicted_pairs if predicted_pairs else 0.,
            'missed_merge_rate':(positive_total-joint_pairs)/positive_total if positive_total else 0.,
            'fragmentation_rate':fragmented/len(expected_sizes),'over_merge_rate':overmerged/len(predicted_sizes),
            'bcubed_precision':precision,'bcubed_recall':recall,'bcubed_f1':2*precision*recall/(precision+recall),
            'ari':float(adjusted_rand_score([expected[b.block_id] for b in blocks],[predicted[b.block_id] for b in blocks])),
            'type_accuracy':sum(predicted_types[k]==expected_types[k] for k in expected)/n,
            'timings_ms':resolution.metrics})
    return {'documents':results,'dataset_kind':'supplied_document_labels'}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("input", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument('--documents',action='store_true'); parser.add_argument('--model-package',type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.documents:
        if args.model_package:
            from app.text_processing.classifiers.lightgbm_classifier import LightGBMRelationshipClassifier
            from app.text_processing.feature_extractor import feature_schema
            factory = lambda:LightGBMRelationshipClassifier(args.model_package,feature_schema())
        else:
            from app.text_processing.classifiers.clustering_classifier import ClusteringRelationshipClassifier
            factory = ClusteringRelationshipClassifier
        result = evaluate_documents(rows,factory)
    else:
        result = evaluate_scenarios(rows)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
