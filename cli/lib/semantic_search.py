import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        # Load the model (downloads automatically the first time)
        self.embeddings_cache_file = "cache/movie_embeddings.npy"
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
    def search(self, query, limit=5):
        if self.embeddings is None: 
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        similarity = []
        for i in range(len(self.documents)):
            doc = self.documents[i]
            emb = self.embeddings[i]
            sim = cosine_similarity(emb, query_embedding)
            similarity.append((sim, doc))
        similarity.sort(key=lambda e:e[0], reverse=True)
        limited = similarity[:limit]
        result = []
        for e in limited:
            result.append({ 'score':e[0], 'title':e[1]['title'], 'description':e[1]['description'] })
        return result

    def generate_embedding(self, text):
        if len(text) == 0 or text.isspace(): 
            raise ValueError("generate_embedding expects non empty and non whitespace text") 

        embeddings = self.model.encode([text])
        embedding = embeddings[0]
        return embedding

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = { d["id"]: d for d in documents }
        documents = [f"{d['title']}: {d['description']}" for d in documents]
        self.embeddings = [self.model.encode(d, show_progress_bar = True) for d in documents]
        np.save(self.embeddings_cache_file, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = { d["id"]: d for d in documents }
        if os.path.exists(self.embeddings_cache_file):
            self.embeddings = np.load(self.embeddings_cache_file)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
        
def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def load_movies():
    movies_json_file  = "data/movies.json"
    with open(movies_json_file) as f: 
        documents = json.load(f)["movies"] # [ { "id":number, "title":string, "description":string },..]
    return documents

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")