
import os
import json
import string
import math
import pickle 
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75


class KeywordSearch:

    def __init__(self):
        self.__index_cache_file  = "cache/index.pkl"
        self.__docmap_cache_file = "cache/docmap.pkl"
        self.__term_frequencies_cache_file = "cache/term_frequencies.pkl"
        self.__doc_lengths_cache_file = "cache/doc_lengths.pkl"
        
        self.__movies_json_file  = "data/movies.json"
        self.__stopwords_file    = "data/stopwords.txt"

        self.__stopwords = self.__load_stopwords()
        self.__movies = self.__load_movies()
        
        self.index = defaultdict(set)   # tokens -> document IDs
        self.docmap = {}  # document IDs -> documents
        self.term_frequencies = {}  # document IDs -> Counter
        self.doc_lengths = {}

    def __load_stopwords(self): 
        with open(self.__stopwords_file) as f: 
            return f.read().split()

    def __load_movies(self):
        with open(self.__movies_json_file) as f: 
            return json.load(f)["movies"] # [ { "id":number, "title":string, "description":string },..]

    def __tokenize(self, text):
        stemmer = PorterStemmer()
        translation = str.maketrans("", "", string.punctuation)
        translated = text.lower().translate(translation)
        words = translated.split()
        words = [w for w in words if w] 
        result = [stemmer.stem(w) for w in words if w not in self.__stopwords]
        result.sort()
        return result

    def __add_document(self, doc_id, text):
        tokens = self.__tokenize(text)
        tokens.sort()

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for t in set(tokens):
            self.index[t].add(doc_id)
        
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        
    def __get_avg_doc_length(self):
        if len(self.doc_lengths) == 0: 
            return 0.0
        else:
            return sum(self.doc_lengths.values()) / len(self.doc_lengths)


    def get_documents(self, term):
        return sorted(list(self.index.get(term, set())))


    def get_tf(self, doc_id, term):
        query = self.__tokenize(term)
        if len(query) != 1: raise ValueError("get_tf expects single token query")
        query = query[0]
        if int(doc_id) not in self.term_frequencies: raise ValueError("doc_id not found")
        return self.term_frequencies[int(doc_id)][query]

    def get_idf(self, term):
        query = self.__tokenize(term)
        if len(query) != 1: raise Exception("get_idf expects single token query")
        query = query[0]

        doc_count = len(self.docmap)
        term_docs = self.get_documents(query)
        term_doc_count = len(term_docs)
        idf = math.log((doc_count + 1) / (term_doc_count + 1))
        return idf

    def get_tfidf(self, doc_id, term):
        tfidf = self.get_tf(doc_id, term) * self.get_idf(term)
        return tfidf


    def get_bm25_idf(self, term):
        query = self.__tokenize(term)
        if len(query) != 1: raise Exception("get_bm25_idf expects single token query")
        query = query[0]

        doc_count = len(self.docmap)
        term_docs = self.get_documents(query)
        term_doc_count = len(term_docs)
        bm25_idf = math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit=5):
        tokens = self.__tokenize(query)
        scores = []

        for doc_id in self.docmap:
            score = 0.0
            for t in tokens:
                score += self.bm25(doc_id, t)
            
            scores.append({
                "id"        : doc_id,
                "title"     : self.docmap[doc_id]['title'],
                "document"  : self.docmap[doc_id]['description'][:100],
                "score"     : score,
            })
            
        scores.sort(reverse=True, key=lambda s: s["score"])
        return scores[:limit]


    def build(self):
        for m in self.__movies:
            document = f"{m['title']} {m['description']}"
            self.docmap[m["id"]] = m
            self.__add_document(m["id"], document)
    
    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open(self.__index_cache_file, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.__docmap_cache_file, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.__term_frequencies_cache_file, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.__doc_lengths_cache_file, "wb") as f:
            pickle.dump(self.doc_lengths, f)
        
    def load(self):
        with open(self.__index_cache_file, "rb") as f:
            self.index = pickle.load(f)
        with open(self.__docmap_cache_file, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.__term_frequencies_cache_file, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.__doc_lengths_cache_file, "rb") as f:
            self.doc_lengths = pickle.load(f)
        return self.index and self.docmap and self.term_frequencies and self.doc_lengths

    def load_or_create(self):
        if not self.load():
            self.build()
            self.save()
            self.load()