import os
import time
import json
import lib.gemini as gemini
from .keyword_search import KeywordSearch
from .chunked_semantic_search import ChunkedSemanticSearch
from .repeat_decorator import repeat_decorator


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.documents_map = {d['id']:d for d in documents}
        self.css = ChunkedSemanticSearch()
        self.css.load_or_create_chunk_embeddings(documents)
        self.ks = KeywordSearch()
        self.ks.load_or_create()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        ss_result = self.css.search_chunks(query, limit * 500)
        ks_result = self.ks.bm25_search(query, limit * 500)
        ss_scores = normalize([s['score'] for s in ss_result])
        ks_scores = normalize([s['score'] for s in ks_result])
        
        scores = {}
        for i in range(len(ss_result)): 
            s = ss_result[i]
            scores[s["id"]] = { 
                "title": s["title"], 
                "description": s["document"],
                "document": self.documents_map[s['id']],
                "semantic_score": ss_scores[i],
                "keyword_score": 0
            }
        for i in range(len(ks_result)):
            s = ks_result[i]
            if s["id"] not in scores: 
                scores[s["id"]] = { 
                    "title": s["title"],
                    "description": s["document"], 
                    "document": self.documents_map[s['id']],
                    "semantic_score": 0
                }
            scores[s["id"]]["keyword_score"] = ks_scores[i]   

        for id in list(scores):
            sss = scores[id].get("semantic_score", 0)
            kss = scores[id].get("keyword_score", 0)
            scores[id]["hybrid_score"] = hybrid_score(kss, sss, alpha)

        result = sorted(scores.items(), reverse=True, key=lambda e: e[1]["hybrid_score"])
        result = list(result)[:limit]
        return dict(result)

    def rrf_search(self, query, k=60, limit=5):
        ss_result = self.css.search_chunks(query, limit * 100)
        ks_result = self.ks.bm25_search(query, limit * 100)
        ss_scores = [rrf_score(i,k) for i in range(len(ss_result))]
        ks_scores = [rrf_score(i,k) for i in range(len(ks_result))]

        scores = {}
        for i in range(len(ss_result)): 
            s = ss_result[i]
            scores[s["id"]] = { 
                "title": s["title"], 
                "description": s["document"],
                "document": self.documents_map[s['id']],
                "semantic_score": ss_scores[i],
                "keyword_score": 0
            }

        for i in range(len(ks_result)):
            s = ks_result[i]
            if s["id"] not in scores: 
                scores[s["id"]] = { 
                    "title": s["title"],
                    "description": s["document"], 
                    "document": self.documents_map[s['id']],
                    "semantic_score": 0
                }
            scores[s["id"]]["keyword_score"] = ks_scores[i]   

        for id in list(scores):
            sss = scores[id].get("semantic_score", 0)
            kss = scores[id].get("keyword_score", 0)
            scores[id]["rrf_score"] = kss + sss

        result = sorted(scores.items(), reverse=True, key=lambda e: e[1]["rrf_score"])
        result = list(result)[:limit]
        return dict(result)


LLM_REQUEST_REPEATS = 3
LLM_REQUEST_PAUSE = 2

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_fix_spelling(query):
    contents =  "Fix any spelling errors in this movie search QUERY.\n" +\
                "No need for some program or script, just FIX the SPELLING ERRORS IN QUERY." +\
                "Only correct obvious typos. Don't change correctly spelled words.\n" +\
                "If no errors, return the ORIGINAL QUERY.\n" +\
                "RETURN ONLY FIXED OR ORIGINAL QUERY. FIXED QUERY SHOULD NOT ADD ANY WORDS, JUST FIX TYPOS\n" +\
                f"Query: '{query}'.\n"
    response = gemini.request(contents)
    return response["response_text"]

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_rewrite_query(query):
    contents =  "Rewrite this movie search query to be more specific and searchable.\n" + \
                "\n" + \
                f"Original: '{query}'\n" + \
                "\n" + \
                "Consider:\n" + \
                "- Common movie knowledge (famous actors, popular films)\n" + \
                "- Genre conventions (horror = scary, animation = cartoon)\n" + \
                "- Keep it concise (under 10 words)\n" + \
                "- It should be a google style search query that's very specific\n" + \
                "- Don't use boolean logic\n" + \
                "\n" + \
                "Examples:\n" + \
                "\n" + \
                "- 'that bear movie where leo gets attacked' -> 'The Revenant Leonardo DiCaprio bear attack'\n" + \
                "- 'movie about bear in london with marmalade' -> 'Paddington London marmalade'\n" + \
                "- 'scary movie with bear from few years ago' -> 'bear horror movie 2015-2020'\n" + \
                "\n" + \
                "Rewritten query:"
    response = gemini.request(contents)
    return response["response_text"]

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_expand_query(query):
    contents = "Expand this movie search query with related terms.\n" + \
                "\n" + \
                "Add synonyms and related concepts that might appear in movie descriptions.\n" + \
                "Keep expansions relevant and focused.\n" + \
                "This will be appended to the original query.\n" + \
                "\n" + \
                "Examples:\n" + \
                "\n" + \
                "- 'scary bear movie' -> 'scary horror grizzly bear movie terrifying film'\n" + \
                "- 'action movie with bear' -> 'action thriller bear chase fight adventure'\n" + \
                "- 'comedy with bear' -> 'comedy funny bear humor lighthearted'\n" + \
                "\n" + \
                f"Query: '{query}'\n"
    response = gemini.request(contents)
    return response["response_text"]

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_rank_query(query, doc):
    contents = "Rate how well this movie matches the search query.\n" +\
                "\n" +\
                f"Query: '{query}'\n" +\
                f"Movie: {doc.get('title', '')} - {doc.get('description', '')}\n" +\
                "\n" +\
                "Consider:\n" +\
                "- Direct relevance to query\n" +\
                "- User intent (what they're looking for)\n" +\
                "- Content appropriateness\n" +\
                "\n" +\
                "Rate 0-10 (10 = perfect match).\n" +\
                "Give me ONLY the number in your response, no other text or explanation.\n" +\
                "\n" +\
                "Score:"
    response = gemini.request(contents)
    response = response["response_text"]
    try: return int(response)
    except (ValueError, TypeError): return 0

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_batch_rank_query(query, doc_list):
    doc_list_str = []
    contents = "Rank these movies by relevance to the search query.\n" +\
                "\n" +\
                f"Query: '{query}'\n" +\
                "\n" +\
                "Movies:\n"
    
    for doc in doc_list:
        contents += f"    Movie: {doc.get('title', '')} - {doc.get('description', '')}\n"
    
    contents += "\nReturn ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:\n" +\
                "[75, 12, 34, 2, 1]\n"
    response = gemini.request(contents)
    json_rsp = json.loads(response["response_text"])
    if len(json_rsp) != len(doc_list): raise ValueError(f"Incorrect response scores list length, list {json_rsp}")
    return json_rsp

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_evaluate_result(query, result):
    formatted_results = [f"  Movie: {r['title']} - {r['document']['description']}" for r in result.values()]

    contents = "Rate how relevant each result is to this query on a 0-3 scale:\n" + \
                "\n" + \
                f"Query: '{query}'\n" + \
                "\n" + \
                "Results:\n" + \
                f"{chr(10).join(formatted_results)}\n" + \
                "\n" + \
                "Scale:\n" + \
                "- 3: Highly relevant\n" + \
                "- 2: Relevant\n" + \
                "- 1: Marginally relevant\n" + \
                "- 0: Not relevant\n" + \
                "\n" + \
                "Do NOT give any numbers out than 0, 1, 2, or 3.\n" + \
                "\n" + \
                "Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:\n" + \
                "[2, 0, 3, 2, 0, 1]\n"
    
    response = gemini.request(contents)
    json_rsp = json.loads(response["response_text"])
    if len(json_rsp) != len(result): raise ValueError(f"Incorrect response scores list length, list {json_rsp}")
    
    i = 0
    for r in result.values():
        r['evaluation'] = json_rsp[i]
        i += 1

    return result

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_rag(query, result):
    formatted_results = [f"  Movie: {r['title']} - {r['document']['description']}" for r in result.values()]

    contents = "Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.\n" +\
                "\n" +\
                f"Query: {query}\n" +\
                "\n" +\
                "Documents:\n" +\
                f"{formatted_results}\n" +\
                "\n" +\
                "Provide a comprehensive answer that addresses the query:\n"
    
    response = gemini.request(contents)
    return response["response_text"]

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_summarize(query, result):
    contents = "Provide information useful to this query by synthesizing information from multiple search results in detail.\n" +\
                "The goal is to provide comprehensive information so that users know what their options are.\n" +\
                "Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.\n" +\
                "This should be tailored to Hoopla users. Hoopla is a movie streaming service.\n" +\
                f"Query: {query}\n" +\
                "Search Results:\n" +\
                f"{result}\n" +\
                "Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:\n"
    
    response = gemini.request(contents)
    return response["response_text"]

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_citations(query, result):
    documents = [f"  Movie: {r['title']} - {r['document']['description']}" for r in result.values()]

    contents = "Answer the question or provide information based on the provided documents.\n" +\
                "\n" +\
                "This should be tailored to Hoopla users. Hoopla is a movie streaming service.\n" +\
                "\n" +\
                "If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.\n" +\
                "\n" +\
                f"Query: {query}\n" +\
                "\n" +\
                "Documents:\n" +\
                f"{documents}\n" +\
                "\n" +\
                "Instructions:\n" +\
                "- Provide a comprehensive answer that addresses the query\n" +\
                "- Cite sources using [1], [2], etc. format when referencing information\n" +\
                "- If sources disagree, mention the different viewpoints\n" +\
                "- If the answer is not in the documents, say 'I do not have enough information'\n" +\
                "- Be direct and informative\n" +\
                "\n" +\
                "Answer:\n"
    
    response = gemini.request(contents)
    return response["response_text"]

@repeat_decorator(LLM_REQUEST_REPEATS, LLM_REQUEST_PAUSE)
def llm_question(question, result):
    documents = [f"  Movie: {r['title']} - {r['document']['description']}" for r in result.values()]

    contents = "Answer the user's question based on the provided movies that are available on Hoopla.\n" +\
                "\n" +\
                "This should be tailored to Hoopla users. Hoopla is a movie streaming service.\n" +\
                "\n" +\
                f"Question: {question}\n" +\
                "\n" +\
                "Documents:\n" +\
                f"{documents}\n" +\
                "\n" +\
                "Instructions:\n" +\
                "- Answer questions directly and concisely\n" +\
                "- Be casual and conversational\n" +\
                "- Don't be cringe or hype-y\n" +\
                "- Talk like a normal person would in a chat conversation\n" +\
                "\n" +\
                "Answer:"
    
    response = gemini.request(contents)
    return response["response_text"]


def llm_fix_query(query, enhance):
    fixed_query = query
    if enhance == 'spell'  : fixed_query = llm_fix_spelling(query)
    if enhance == 'rewrite': fixed_query = llm_rewrite_query(query)
    if enhance == 'expand' : fixed_query = llm_expand_query(query)
    return fixed_query

def llm_rerank(result, query, limit):
    i = 1
    for [id, s] in result.items():
        print(f" Reranking {i}. id({id}) {s['title']}", end="")
        i += 1
        s["reranked_score"] = llm_rank_query(query, s['document'])
        print(f"    Reranked score: {s['reranked_score']}")
        time.sleep(1)
    print("Reranked.")
    result = sorted(result.items(), reverse=True, key=lambda e: e[1]["reranked_score"])
    result = list(result)[:limit]
    result = dict(result)
    return result

def llm_batch_rerank(result, query, limit):
    doc_list = []
    for [id, s] in result.items():
        doc_list.append(s['document'])

    scores = llm_batch_rank_query(query, doc_list)
    i = 0
    for [id, s] in result.items():
        s["reranked_score"] = scores[i]
        i += 1

    print("Reranked.")
    result = sorted(result.items(), reverse=True, key=lambda e: e[1]["reranked_score"])
    result = list(result)[:limit]
    result = dict(result)
    return result


def rrf_score(rank, k=60):
    return 1 / (k + rank)

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize(value_list):
    _min = min(value_list)
    _max = max(value_list)
    d = _max - _min
    if d == 0: return [1.0]*len(value_list)
    return [(v-_min)/d for v in value_list]