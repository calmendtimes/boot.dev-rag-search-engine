import os
import re
import json
import numpy as np
import lib.semantic_search as ss


class ChunkedSemanticSearch(ss.SemanticSearch):

    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings_cache_file = "cache/chunk_embeddings.npy"
        self.chunk_metadata_file = "cache/chunk_metadata.json"
        self.chunks = None
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None: 
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        
        qemb = super().generate_embedding(query)
        chunk_scores = []
        movie_scores = {}
        
        for i in range(len(self.chunk_embeddings)):
            emb = self.chunk_embeddings[i]
            sim = ss.cosine_similarity(emb, qemb)
            meta = self.chunk_metadata[i]
            cidx = meta["chunk_idx"]
            midx = meta["movie_idx"]
            
            chunk_scores.append({
                "chunk_idx": cidx, # chunk_idx: The index of the chunk within the document
                "movie_idx": midx, # movie_idx: The index of the document in self.documents (you'll need to use self.chunk_metadata to map back to this)
                "score": sim       # score: The cosine similarity score
            })
            
            if midx not in movie_scores or movie_scores[midx] < sim:
                movie_scores[midx] = sim
        
        movie_scores = sorted(movie_scores.items(), key=lambda e:e[1], reverse=True)
        movie_scores = movie_scores[:limit]

        result = []
        for m in movie_scores:
            md = {
                "id"        : self.documents[m[0]]["id"],
                "title"     : self.documents[m[0]]["title"],
                "document"  : self.documents[m[0]]["description"][:100],
                "score"     : round(m[1], 4),
                #"metadata"  : metadata or {}
            }
            result.append(md)
        
        return result

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = { d["id"]: d for d in documents }
        
        self.chunks = []
        self.chunk_metadata = []
        for id in range(len(self.documents)):
            d = self.documents[id]
            if not d["description"]: continue
            dscs = semantic_chunk(d["description"], 4, 1)
            midx = d["id"]

            for isc in range(len(dscs)):
                sc = dscs[isc]
                self.chunks.append(sc)
                self.chunk_metadata.append({
                    "movie_idx": id,         # The index of the document in self.documents
                    "chunk_idx": isc,        # The index of the chunk within the document
                    "total_chunks": len(dscs)
                })
        
        self.chunk_embeddings = [self.model.encode(c, show_progress_bar = True) for c in self.chunks]
        np.save(self.chunk_embeddings_cache_file, self.chunk_embeddings)
        with open(self.chunk_metadata_file, 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(self.chunks)}, f, indent=2)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = { d["id"]: d for d in documents }
        if os.path.exists(self.chunk_metadata_file):
            with open(self.chunk_metadata_file) as f:
                self.chunk_metadata = json.load(f)["chunks"]
        if os.path.exists(self.chunk_embeddings_cache_file):
            self.chunk_embeddings = np.load(self.chunk_embeddings_cache_file)
            if len(self.chunk_embeddings) == len(self.chunk_metadata):
                return self.chunk_embeddings
            else: 
                print(f"Embedding and Metadata count mismatch {len(self.chunk_embeddings)} {len(self.chunk_metadata)}")

        return self.build_chunk_embeddings(documents)



def chunk(text, chunk_size, overlap):
    tokens = text.split()
    k = chunk_size
    o = overlap
    return [' '.join(tokens[i:i+k]) for i in range(0, len(tokens), k-o) if i+o<len(tokens)]

def semantic_chunk(text, chunk_sentence_count, overlap):
    text = text.strip()
    if not text: return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and sentences[0][-1] not in '.?!': [text]

    sentences = [s.strip() for s in sentences if s.strip()]

    [k, o, ls] = [chunk_sentence_count, overlap, len(sentences)]
    return [' '.join(sentences[i:i+k]) for i in range(0, ls, k-o) if i+o<ls]