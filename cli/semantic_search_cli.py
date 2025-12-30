#!/usr/bin/env python3

import argparse
import re
import lib.semantic_search as SS
import lib.chunked_semantic_search as CSS


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
    chunk_parser = subparsers.add_parser("chunk", help="chunk <text> in to [--chunk-size] token count parts")
    chunk_parser.add_argument("text", type=str, help="text")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="chunk tokens count")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="overlap")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="chunk <text> in to 'sentences' up to [--max_chunk-size] parts")
    semantic_chunk_parser.add_argument("text", type=str, help="text")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="text")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="text")
    embed_text_parser = subparsers.add_parser("embed_chunks", help="Generate movies chunks embeddings")
    search_chunked_parser = subparsers.add_parser("search_chunked", help="search <text> in chunked movies")
    search_chunked_parser.add_argument("text", type=str, help="text")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="number of results")
    

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
        case "chunk":
            chunks = CSS.chunk(args.text, args.chunk_size, args.overlap)
            print(F"Chunking {len(args.text)} characters")
            for i in range(len(chunks)): print(f"{i+1}. {chunks[i]}")
        case "semantic_chunk":
            chunks = CSS.semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(F"Semantically chunking {len(args.text)} characters")
            for i in range(len(chunks)): print(f"{i+1}. {chunks[i]}")
        case "embed_chunks":
            documents = SS.load_movies()
            css = CSS.ChunkedSemanticSearch()
            chunk_embeddings = css.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(chunk_embeddings)} chunked embeddings")
        case "search_chunked":
            documents = SS.load_movies()
            css = CSS.ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(documents)
            result = css.search_chunks(args.text, args.limit)
            for i in range(len(result)):
                r = result[i]
                print(f"\n{i+1}.  {r['title']} (score: {r['score']:.4f})")
                print(f"    {r['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()