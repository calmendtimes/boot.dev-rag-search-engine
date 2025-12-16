#!/usr/bin/env python3

import argparse
import lib.semantic_search as SS

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verify Semantic Search model")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verify movies.json embeddings")
    embed_text_parser = subparsers.add_parser("embed_text", help="Generate text embedding <text>")
    embed_text_parser.add_argument("text", type=str, help="text")
    embedquery_parser = subparsers.add_parser("embedquery", help="Generate text embedding <text>")
    embedquery_parser.add_argument("text", type=str, help="text")
    search_parser = subparsers.add_parser("search", help="search <text> in movies")
    search_parser.add_argument("text", type=str, help="text")
    search_parser.add_argument("--limit", type=int, default=5, help="number of results")
    

    args = parser.parse_args()

    match args.command:
        case "verify":
            SS.verify_model()
        case "verify_embeddings":
            SS.verify_embeddings()
        case "embed_text":
            ss = SS.SemanticSearch()
            embedding = ss.generate_embedding(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")
        case "embedquery":
            SS.embed_query_text(args.text)
        case "search":
            documents = SS.load_movies()
            ss = SS.SemanticSearch()
            ss.load_or_create_embeddings(documents)
            result = ss.search(args.text, args.limit)
            for i in range(len(result)):
                r = result[i]
                print(f"{i+1}.  {r['title']} (score: {r['score']:.2f})")
                print(f"    {r['description'][:55]}...")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()