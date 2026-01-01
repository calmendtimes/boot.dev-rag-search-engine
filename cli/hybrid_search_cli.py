import argparse
import lib.semantic_search as SS
import lib.hybrid_search as HS

def print_weighted_search(result):
    i = 1
    for [id, s] in result.items():
        print(f" {i}. id({id}) {s['title']}")
        print(f"    Hybrid Score: {s['hybrid_score']:.4f}")
        print(f"    BM25: {s['keyword_score']:.4f},    Semantic: {s['semantic_score']:.4f}")
        print(f"    {s['document']}")
        i += 1

def print_rrf_search(result, limit):
    i = 1
    for [id, s] in result.items():
        print(f" {i}. id({id}) {s['title']}")
        print(f"    RRF: {s['rrf_score']:.4f}")
        print(f"    BM25: {s['keyword_score']:.4f},    Semantic: {s['semantic_score']:.4f}")
        print(f"    {s['document']}")
        if i == limit: break
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")    
    normalize_parser = subparsers.add_parser("normalize", help="min max normalize list")
    normalize_parser.add_argument("values", type=float, nargs='+', help="list values")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="weighted search of <query> with [--alpha [0,1]] weighting and [--limit N] results.")
    weighted_search_parser.add_argument("query", type=str, help="Query to get weighted search results for.")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=0.5, help="weight of exact matching vs embedding matching")
    weighted_search_parser.add_argument("--limit", type=int,   nargs='?', default=5, help="Number of results")
    rrf_search_parser = subparsers.add_parser("rrf-search", help="weighted search of <query> with [--alpha [0,1]] weighting and [--limit N] results.")
    rrf_search_parser.add_argument("query", type=str, help="Query to get weighted search results for.")
    rrf_search_parser.add_argument("-k", type=int, nargs='?', default=1, help="rrf k parameter")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = HS.normalize(args.values)
            for n in normalized: print(f"* {n:.4f}")   
        case "weighted-search":
            documents = SS.load_movies()
            hs = HS.HybridSearch(documents)
            result = hs.weighted_search(args.query, args.alpha, args.limit)
            print_weighted_search(result)  
        case "rrf-search":
            documents = SS.load_movies()
            hs = HS.HybridSearch(documents)
            fixed_query = ""
            if args.enhance == 'spell': 
                fixed_query = HS.fix_spelling(args.query)
            if args.enhance == 'rewrite': 
                fixed_query = HS.rewrite_query(args.query)
            if args.enhance == 'expand': 
                fixed_query = HS.expand_query(args.query)
            if fixed_query and fixed_query != args.query:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{fixed_query}'\n")
                result = hs.rrf_search(fixed_query, args.k, args.limit)
                print_rrf_search(result, args.limit)
            else:
                result = hs.rrf_search(args.query, args.k, args.limit)
                print_rrf_search(result, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()  