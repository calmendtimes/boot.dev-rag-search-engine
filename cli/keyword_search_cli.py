#!/usr/bin/env python3

import argparse
from inverted_index import InvertedIndex


def print_search_result(result):
    for id, doc in result.items():
        print(f"{id}. Movie Title {doc['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build inverted index for movies")
    tf_parser = subparsers.add_parser("tf", help="<term> frequency in document <doc_id>")
    tf_parser.add_argument("doc_id", type=str, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search query")
    idf_parser = subparsers.add_parser("idf", help="inverse document frequency <term>")
    idf_parser.add_argument("term", type=str, help="Search query")
    tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF of <term> in <doc_id>")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search query")

    args = parser.parse_args()  

    index = InvertedIndex()

    match args.command:
        case "search":
            try:
                index.load()  
                found = index.get_documents(args.query)
                print_search_result(found)
            except Exception as e:
                print("Error", e)
                print("Exiting application")
        case "tf":
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(tf)
        case "idf":
            index.load()
            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index.load()
            tf_idf = index.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "build":
            index.build()
            index.save()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
