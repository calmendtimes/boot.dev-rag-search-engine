
import os
import json
import string
import math
import pickle 
from collections import Counter
from nltk.stem import PorterStemmer

class InvertedIndex:

    def __init__(self):
        self.__index_cache_file  = "cache/index.pkl"
        self.__docmap_cache_file = "cache/docmap.pkl"
        self.__term_frequencies_cache_file = "cache/term_frequencies.pkl"
        self.__movies_json_file  = "data/movies.json"
        self.__stopwords_file    = "data/stopwords.txt"

        self.__stopwords = self.__load_stopwords()
        self.__movies = self.__load_movies()
        
        self.index = {}   # tokens -> document IDs
        self.docmap = {}  # document IDs -> documents
        self.term_frequencies  = {}  # document IDs -> Counter

    def __load_stopwords(self): 
        with open(self.__stopwords_file) as f: 
            return f.read().split()

    def __load_movies(self):
        with open(self.__movies_json_file) as f: 
            return json.load(f)["movies"] # [ { "id":number, "title":string, "description":string },..]

    def __denoise(self, text):
        stemmer = PorterStemmer()
        translation = str.maketrans("", "", string.punctuation)
        translated = text.lower().translate(translation)
        words = translated.split()
        words = [w for w in words if w]
        stems = {stemmer.stem(w) for w in words if w not in self.__stopwords}
        result = list(stems)
        result.sort()
        return result

    def __add_document(self, doc_id, text):
        tokens = self.__denoise(text)
        tokens.sort()

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for t in tokens:
            if t not in self.index: 
                self.index[t] = set()
            self.index[t].add(doc_id)
            self.term_frequencies[doc_id][t] = self.docmap[doc_id].count(t)
        
    def get_documents(self, term):
        docs = set()
        query = self.__denoise(term)

        for q in query:
            if q in self.index:
                docs.update(self.index[q])

        result = {id: self.docmap[id] for id in docs}
        return result

    def get_tf(self, doc_id, term):
        query = self.__denoise(term)
        if len(query) != 1: raise ValueError("get_tf expects single token query")
        if int(doc_id) not in self.term_frequencies: raise ValueError("doc_id not found")
        return self.term_frequencies[int(doc_id)][query[0]]

    def get_idf(self, term):
        doc_count = len(self.docmap)
        term_docs = self.get_documents(term)
        term_doc_count = len(term_docs)
        idf = math.log((doc_count + 1) / (term_doc_count + 1))
        return idf

    def get_tfidf(self, doc_id, term):
        tfidf = self.get_tf(doc_id, term) * self.get_idf(term)
        return tfidf

    def build(self):
        for m in self.__movies:
            document = f"{m['title']} {m['description']}"
            self.docmap[m["id"]] = document
            self.__add_document(m["id"], document)
    
    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open(self.__index_cache_file, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.__docmap_cache_file, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.__term_frequencies_cache_file, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        
    def load(self):
        with open(self.__index_cache_file, "rb") as f:
            self.index = pickle.load(f)
        with open(self.__docmap_cache_file, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.__term_frequencies_cache_file, "rb") as f:
            self.term_frequencies = pickle.load(f)

