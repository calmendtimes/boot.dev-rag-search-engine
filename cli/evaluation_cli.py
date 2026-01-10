import argparse
import json
import lib.hybrid_search as HS
import lib.semantic_search as SS


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json") as f: 
        golden_dataset = json.load(f) # { "test_cases": [ { "query": "q", "relevant_docs": ["d1", "d2"] }, ] }
        test_cases = golden_dataset["test_cases"]

    documents = SS.load_movies()
    hs = HS.HybridSearch(documents)
    
    print(f"k={limit}")
    for tc in test_cases:
        result = hs.rrf_search(tc['query'], 60, limit)
        expected = tc['relevant_docs']
        received = [r['title'] for r in result.values()]
        intersection = list(set(expected) & set(received))
        precision = len(intersection) / limit
        recall = len(intersection) / len(expected)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"- Query: {tc['query']}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: " + ", ".join(received))
        print(f"  - Relevant: " + ", ".join(expected))
        print(f"  - Relevant Retrieved: " + ", ".join(intersection))



if __name__ == "__main__":
    main()