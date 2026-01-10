import argparse
import lib.hybrid_search as HS
import lib.semantic_search as SS


def print_rag(result, response):
    print("Search Results:")
    for r in result.values(): print(f"  - {r['title']}")
    print("RAG Response:")
    print(response)

def print_summary(result, response):
    print("Search Results:")
    for r in result.values(): print(f"  - {r['title']}")
    print("RAG Summary:")
    print(response)

def print_answer(result, response):
    print("Search Results:")
    for r in result.values(): print(f"  - {r['title']}")
    print("Answer:")
    print(response)



def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser("summarize", help="Summarization of limit number of search results <query> [--limit].")
    summarize_parser.add_argument("query", type=str, help="Query to get weighted search results for.")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results")
    citations_parser = subparsers.add_parser("citations", help="Search <query> with citations, return [--limit=5] results")
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results")
    question_parser = subparsers.add_parser("question", help="Ask a <question> about movies from [--limit=5] results")
    question_parser.add_argument("question", type=str, help="question")
    question_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results")
    
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            documents = SS.load_movies()
            hs = HS.HybridSearch(documents)
            result = hs.rrf_search(query) 
            response = HS.llm_rag(query, result)      
            print_rag(result, response)   
        case "summarize":
            query = args.query
            documents = SS.load_movies()
            hs = HS.HybridSearch(documents)
            result = hs.rrf_search(query, limit=args.limit)
            response = HS.llm_summarize(query, result)
            print_summary(result, response)
        case "citations":
            query = args.query
            documents = SS.load_movies()
            hs = HS.HybridSearch(documents)
            result = hs.rrf_search(query, limit=args.limit)
            response = HS.llm_citations(query, result) 
            print_answer(result, response)
        case "question":
            question = args.question
            documents = SS.load_movies()
            hs = HS.HybridSearch(documents)
            result = hs.rrf_search(question, limit=args.limit)
            response = HS.llm_question(question, result) 
            print_answer(result, response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()