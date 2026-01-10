import os
import torch
import torch.nn.functional as torchF
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import lib.semantic_search as SS


class MultimodalSearch:

    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.embeddings_cache_file = "cache/multimodal_text_embeddings.npy"
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{d['title']}: {d['description']}" for d in documents]
        self.texts = self.texts
        self.load_or_build_embeddings()

    def load_or_build_embeddings(self):
        if os.path.exists(self.embeddings_cache_file):
            self.text_embeddings = np.load(self.embeddings_cache_file)
            if len(self.text_embeddings) == len(self.texts): 
                return self.text_embeddings
            else: 
                print(f"Embedding and Texts count mismatch {len(self.text_embeddings)} {len(self.texts)}")
                print("Rebuilding...")
                
        self.text_embeddings = self.model.encode_query(self.texts, show_progress_bar=True)
        np.save(self.embeddings_cache_file, self.text_embeddings)
        print("Rebuilt.")


    def embed_image(self, path):
        image = Image.open(path)
        embedding = self.model.encode([image])
        return embedding[0] 
    
    def search_with_image(self, path):
        ie = self.embed_image(path)
        cos_sim = []
        for i in range(len(self.texts)):
            cos_sim.append((SS.cosine_similarity(ie, self.text_embeddings[i]), self.documents[i])) 
        cos_sim.sort(key=lambda e: e[0], reverse=True) 
        return cos_sim[:5]
        