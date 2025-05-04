from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Any
import torch
import gc
import os
import psutil
import json

class FAQEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the FAQ embedder with a sentence transformer model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding model using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.faqs = None
        self.embeddings = None
    
    def create_embeddings(self, faqs: List[Dict[str, Any]], batch_size: int = None) -> None:
        """
        Create embeddings for all FAQs and build FAISS index
        """
        self.faqs = faqs
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
        batch_size = batch_size or min(64, int(available_memory * 4))
        print(f"Creating embeddings for {len(faqs)} FAQs in batches of {batch_size}...")
        
        questions = [faq['question'] for faq in faqs]
        all_embeddings = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
            batch_embeddings = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(all_embeddings).astype('float32')
        all_embeddings = None
        gc.collect()
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        print(f"Created embeddings of shape {self.embeddings.shape}")
        print(f"FAISS index contains {self.index.ntotal} vectors")
    
    def retrieve_relevant_faqs(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant FAQs for a given query
        """
        if self.index is None or self.faqs is None or self.embeddings is None:
            raise ValueError("Embeddings not created yet. Call create_embeddings first.")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        relevant_faqs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.faqs):
                faq = self.faqs[idx].copy()
                similarity = 1.0 / (1.0 + distances[0][i])
                faq['similarity'] = similarity
                relevant_faqs.append(faq)
        
        return relevant_faqs
    
    def save(self, path: str):
        """
        Save embeddings and FAQs to disk
        """
        os.makedirs(path, exist_ok=True)
        self.model.save(path)
        faiss.write_index(self.index, f"{path}/index.bin")
        with open(f"{path}/faqs.json", "w") as f:
            json.dump(self.faqs, f)
    
    def load(self, path: str):
        """
        Load embeddings and FAQs from disk
        """
        self.model = SentenceTransformer(path)
        self.index = faiss.read_index(f"{path}/index.bin")
        with open(f"{path}/faqs.json", "r") as f:
            self.faqs = json.load(f)
        self.embeddings = np.array([self.model.encode(faq["question"]) for faq in self.faqs]).astype('float32')