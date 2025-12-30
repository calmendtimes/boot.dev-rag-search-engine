#!/usr/bin/env python3

import argparse
import lib.keyword_search as KS


def print_search_result(result):
    for id, doc in result.items():
        print(f"{id}. Movie Title {doc['title']}")

def print_bm25search_result(result):
    for r in result:
        print(f" ({r['id']}) {r['title']} - {r['score']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build inverted ks for movies")
    tf_parser = subparsers.add_parser("tf", help="<term> frequency in document <doc_id>")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search query")
    idf_parser = subparsers.add_parser("idf", help="inverse document frequency <term>")
    idf_parser.add_argument("term", type=str, help="Search query")
    tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF of <term> in <doc_id>")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search query")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given <term>")
    bm25_idf_parser.add_argument("term", type=str, help="Search query")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=KS.BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=KS.BM25_B, help="Tunable BM25 b parameter")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()  

    ks = KS.KeywordSearch()

    match args.command:
        case "search":
            try:
                ks.load()  
                found = ks.get_documents(args.query)
                print_search_result(found)
            except Exception as e:
                print("Error", e)
                print("Exiting application")
        case "tf":
            ks.load()
            tf = ks.get_tf(args.doc_id, args.term)
            print(tf)
        case "idf":
            ks.load()
            idf = ks.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            ks.load()
            tf_idf = ks.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            ks.load()
            bm25idf = ks.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            ks.load()
            bm25tf = ks.get_bm25_tf(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            ks.load()
            bm25search = ks.bm25_search(args.query)
            print_bm25search_result(bm25search)
        case "build":
            ks.build()
            ks.save()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
