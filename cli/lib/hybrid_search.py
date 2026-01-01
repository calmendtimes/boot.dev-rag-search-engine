import os
import lib.gemini as gemini
from .keyword_search import KeywordSearch
from .chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
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
                "document": s["document"],
                "semantic_score": ss_scores[i],
                "keyword_score": 0
            }
        for i in range(len(ks_result)):
            s = ks_result[i]
            if s["id"] not in scores: 
                scores[s["id"]] = { 
                    "title": s["title"],
                    "document": s["document"], 
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

    def rrf_search(self, query, k, limit=5):
        ss_result = self.css.search_chunks(query, limit * 500)
        ks_result = self.ks.bm25_search(query, limit * 500)
        ss_scores = [rrf_score(i,k) for i in range(len(ss_result))]
        ks_scores = [rrf_score(i,k) for i in range(len(ks_result))]

        scores = {}
        for i in range(len(ss_result)): 
            s = ss_result[i]
            scores[s["id"]] = { 
                "title": s["title"], 
                "document": s["document"],
                "semantic_score": ss_scores[i],
                "keyword_score": 0
            }

        for i in range(len(ks_result)):
            s = ks_result[i]
            if s["id"] not in scores: 
                scores[s["id"]] = { 
                    "title": s["title"],
                    "document": s["document"], 
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


def fix_spelling(query):
    contents =  "Fix any spelling errors in this movie search QUERY.\n" +\
                "No need for some program or script, just FIX the SPELLING ERRORS IN QUERY." +\
                "Only correct obvious typos. Don't change correctly spelled words.\n" +\
                "If no errors, return the ORIGINAL QUERY.\n" +\
                "RETURN ONLY FIXED OR ORIGINAL QUERY. FIXED QUERY SHOULD NOT ADD ANY WORDS, JUST FIX TYPOS\n" +\
                f"Query: '{query}'.\n"
    response = gemini.request(contents)
    return response["response_text"]

def rewrite_query(query):
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

def expand_query(query):
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