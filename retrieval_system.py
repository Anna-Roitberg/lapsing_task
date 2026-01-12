import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MinimalRAG:
    def __init__(self, docs_dir='rag_docs'):
        self.docs_dir = docs_dir
        self.chunks = []
        self.chunk_sources = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        self._load_and_index()
        
    def _load_and_index(self):
        """Loads text files, chunks them, and builds the TF-IDF index."""
        # Support both new structure (md) and legacy (txt)
        file_paths = glob.glob(os.path.join(self.docs_dir, "**", "*.md"), recursive=True) + \
                     glob.glob(os.path.join(self.docs_dir, "*.txt"))
        
        print(f"Loading documents from {self.docs_dir}...")
        for path in file_paths:
            filename = os.path.basename(path)
            
            # Extract citation ID if present (e.g. Doc1_...)
            citation_id = filename.split('_')[0] if filename.startswith("Doc") else filename
            source_label = f"[{citation_id}]" if filename.startswith("Doc") else filename
            
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Simple chunking: Split by double newlines (paragraphs) 
            raw_chunks = text.split('\n\n')
            
            clean_chunks = [c.strip() for c in raw_chunks if c.strip()]
            
            for chunk in clean_chunks:
                # Add source info to the chunk text for context: [Doc1] content...
                full_chunk_text = f"Source: {source_label} ({filename})\n{chunk}"
                self.chunks.append(full_chunk_text)
                self.chunk_sources.append(source_label)
                
        print(f"Indexed {len(self.chunks)} chunks from {len(file_paths)} files.")
        
        # Vectorize
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        else:
            print("Warning: No documents found to index.")

    def retrieve(self, query, k=3):
        """Retrieves top k relevant chunks for the query."""
        if not self.chunks or self.tfidf_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        if k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[-k:][::-1]
            
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.05: # Minimal threshold
                results.append({
                    'chunk': self.chunks[idx],
                    'source': self.chunk_sources[idx],
                    'score': float(score)
                })
                
        return results

if __name__ == "__main__":
    # Test
    rag = MinimalRAG()
    res = rag.retrieve("customer missed payment grace period", k=2)
    for r in res:
        print(f"[{r['score']:.4f}] {r['source']}: {r['chunk'][:50]}...")
