"""
Embedding Store Module for Causal RAG System.
Handles embedding generation, FAISS vector indexing, and similarity search with metadata filtering.
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import hashlib

from config import get_config, EmbeddingConfig, VectorStoreConfig
from data_processor import TurnDocument, create_enriched_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    document: TurnDocument
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "rank": self.rank
        }


class EmbeddingModel:
    """
    Handles embedding generation using sentence-transformers or OpenAI.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedding model."""
        self.config = config or EmbeddingConfig()
        self._model = None
        self._openai_client = None
        
    def _load_sentence_transformer(self):
        """Load sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformer model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
            logger.info("Sentence-transformer model loaded successfully")
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
    
    def _load_openai_client(self):
        """Load OpenAI client."""
        try:
            from openai import OpenAI
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key not provided")
            self._openai_client = OpenAI(api_key=self.config.openai_api_key)
            logger.info("OpenAI client initialized")
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim).
        """
        if self.config.use_openai:
            return self._embed_openai(texts)
        else:
            return self._embed_sentence_transformer(texts)
    
    def _embed_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        if self._model is None:
            self._load_sentence_transformer()
        
        logger.info(f"Embedding {len(texts)} texts with sentence-transformer")
        
        # Process in batches
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.config.normalize_embeddings
        )
        
        return np.array(embeddings)
    
    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        if self._openai_client is None:
            self._load_openai_client()
        
        logger.info(f"Embedding {len(texts)} texts with OpenAI")
        
        all_embeddings = []
        
        # Process in batches (OpenAI has a limit)
        batch_size = min(self.config.batch_size, 100)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._openai_client.embeddings.create(
                model=self.config.model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        embeddings = np.array(all_embeddings)
        
        # Normalize if requested
        if self.config.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            NumPy array of embedding with shape (embedding_dim,).
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self.config.use_openai:
            # text-embedding-3-large has 3072 dimensions
            return 3072
        else:
            if self._model is None:
                self._load_sentence_transformer()
            return self._model.get_sentence_embedding_dimension()


class FAISSVectorStore:
    """
    FAISS-based vector store with metadata filtering support.
    """
    
    def __init__(
        self, 
        embedding_model: Optional[EmbeddingModel] = None,
        config: Optional[VectorStoreConfig] = None
    ):
        """Initialize the FAISS vector store."""
        self.config = config or VectorStoreConfig()
        self.embedding_model = embedding_model or EmbeddingModel()
        
        self._index = None
        self._documents: List[TurnDocument] = []
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}  # transcript_id:turn_id -> index
        
        self._faiss = None
        self._load_faiss()
    
    def _load_faiss(self):
        """Load FAISS library."""
        try:
            import faiss
            self._faiss = faiss
            logger.info("FAISS loaded successfully")
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")
    
    def build_index(self, documents: List[TurnDocument], show_progress: bool = True) -> None:
        """
        Build the FAISS index from documents.
        
        Args:
            documents: List of TurnDocument objects to index.
            show_progress: Whether to show progress bar.
        """
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        logger.info(f"Building FAISS index for {len(documents)} documents")
        
        # Store documents and metadata
        self._documents = documents
        self._metadata = [doc.get_metadata() for doc in documents]
        
        # Build ID mapping
        self._id_to_idx = {}
        for i, doc in enumerate(documents):
            key = f"{doc.transcript_id}:{doc.turn_id}"
            self._id_to_idx[key] = i
        
        # Generate embeddings
        texts = [create_enriched_text(doc) for doc in documents]
        embeddings = self.embedding_model.embed_texts(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        
        if self.config.similarity_metric == "cosine":
            # For cosine similarity, use inner product on normalized vectors
            self._index = self._faiss.IndexFlatIP(dimension)
        elif self.config.similarity_metric == "l2":
            self._index = self._faiss.IndexFlatL2(dimension)
        else:
            self._index = self._faiss.IndexFlatIP(dimension)
        
        # Add vectors to index
        self._index.add(embeddings.astype(np.float32))
        
        logger.info(f"FAISS index built with {self._index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        outcome_filter: Optional[str] = None,
        domain_filter: Optional[str] = None,
        speaker_filter: Optional[str] = None,
        transcript_ids: Optional[List[str]] = None,
        exclude_transcript_ids: Optional[List[str]] = None,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar documents with optional filtering.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            outcome_filter: Filter by outcome (e.g., "Escalation").
            domain_filter: Filter by domain.
            speaker_filter: Filter by speaker ("Agent" or "Customer").
            transcript_ids: Only search within these transcript IDs.
            exclude_transcript_ids: Exclude these transcript IDs.
            min_similarity: Minimum similarity threshold.
            
        Returns:
            List of SearchResult objects.
        """
        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_single(query)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search with more results to allow for filtering
        search_k = min(top_k * 5, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, search_k)
        
        # Apply filters and build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            
            doc = self._documents[idx]
            metadata = self._metadata[idx]
            
            # Apply filters
            if outcome_filter and metadata.get("outcome") != outcome_filter:
                continue
            if domain_filter and metadata.get("domain") != domain_filter:
                continue
            if speaker_filter and metadata.get("speaker") != speaker_filter:
                continue
            if transcript_ids and doc.transcript_id not in transcript_ids:
                continue
            if exclude_transcript_ids and doc.transcript_id in exclude_transcript_ids:
                continue
            if score < min_similarity:
                continue
            
            result = SearchResult(
                document=doc,
                score=float(score),
                rank=len(results) + 1
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_outcome(
        self,
        query: str,
        outcome: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search specifically within an outcome category.
        
        Args:
            query: Search query text.
            outcome: Outcome to filter by.
            top_k: Number of results.
            domain_filter: Optional domain filter.
            
        Returns:
            List of SearchResult objects.
        """
        return self.search(
            query=query,
            top_k=top_k,
            outcome_filter=outcome,
            domain_filter=domain_filter
        )
    
    def get_conversation_turns(self, transcript_id: str) -> List[TurnDocument]:
        """
        Get all turns for a specific transcript.
        
        Args:
            transcript_id: Transcript ID.
            
        Returns:
            List of TurnDocument objects sorted by turn_id.
        """
        turns = [doc for doc in self._documents if doc.transcript_id == transcript_id]
        return sorted(turns, key=lambda x: x.turn_id)
    
    def get_documents_by_outcome(self, outcome: str) -> List[TurnDocument]:
        """Get all documents with a specific outcome."""
        return [doc for doc in self._documents if doc.outcome == outcome]
    
    def get_unique_transcript_ids(self, outcome: Optional[str] = None) -> List[str]:
        """Get unique transcript IDs, optionally filtered by outcome."""
        if outcome:
            docs = self.get_documents_by_outcome(outcome)
        else:
            docs = self._documents
        return list(set(doc.transcript_id for doc in docs))
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the index and metadata to disk.
        
        Args:
            path: Directory to save to. Uses config default if not provided.
        """
        save_path = Path(path or self.config.index_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path / "index.faiss"
        self._faiss.write_index(self._index, str(index_file))
        
        # Save metadata and documents
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'documents': self._documents,
                'metadata': self._metadata,
                'id_to_idx': self._id_to_idx
            }, f)
        
        logger.info(f"Index saved to {save_path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """
        Load the index and metadata from disk.
        
        Args:
            path: Directory to load from. Uses config default if not provided.
        """
        load_path = Path(path or self.config.index_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index directory not found: {load_path}")
        
        # Load FAISS index
        index_file = load_path / "index.faiss"
        self._index = self._faiss.read_index(str(index_file))
        
        # Load metadata and documents
        metadata_file = load_path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self._documents = data['documents']
            self._metadata = data['metadata']
            self._id_to_idx = data['id_to_idx']
        
        logger.info(f"Index loaded from {load_path} with {self._index.ntotal} vectors")
    
    def exists(self, path: Optional[str] = None) -> bool:
        """Check if a saved index exists."""
        load_path = Path(path or self.config.index_path)
        return (load_path / "index.faiss").exists() and (load_path / "metadata.pkl").exists()
    
    @property
    def total_documents(self) -> int:
        """Get total number of indexed documents."""
        return len(self._documents) if self._documents else 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self._documents:
            return {"status": "empty"}
        
        outcome_counts = {}
        domain_counts = {}
        speaker_counts = {}
        
        for doc in self._documents:
            outcome_counts[doc.outcome] = outcome_counts.get(doc.outcome, 0) + 1
            domain_counts[doc.domain] = domain_counts.get(doc.domain, 0) + 1
            speaker_counts[doc.speaker] = speaker_counts.get(doc.speaker, 0) + 1
        
        return {
            "total_documents": len(self._documents),
            "total_vectors": self._index.ntotal if self._index else 0,
            "unique_transcripts": len(set(doc.transcript_id for doc in self._documents)),
            "outcomes": outcome_counts,
            "domains": domain_counts,
            "speakers": speaker_counts
        }


class EmbeddingStore:
    """
    High-level interface for the embedding and vector storage system.
    Combines embedding generation and FAISS indexing.
    """
    
    def __init__(self, config=None):
        """Initialize the embedding store."""
        self.config = config or get_config()
        self.embedding_model = EmbeddingModel(self.config.embedding)
        self.vector_store = FAISSVectorStore(
            embedding_model=self.embedding_model,
            config=self.config.vector_store
        )
        self._initialized = False
    
    def initialize(
        self, 
        documents: List[TurnDocument],
        force_rebuild: bool = False
    ) -> None:
        """
        Initialize the store with documents.
        
        Loads from disk if available, otherwise builds new index.
        
        Args:
            documents: List of TurnDocument objects.
            force_rebuild: Force rebuilding even if index exists.
        """
        index_path = self.config.vector_store.index_path
        
        if not force_rebuild and self.vector_store.exists(index_path):
            logger.info("Loading existing index from disk")
            self.vector_store.load(index_path)
        else:
            logger.info("Building new index")
            self.vector_store.build_index(documents)
            
            # Save for future use
            if self.config.enable_caching:
                self.vector_store.save(index_path)
        
        self._initialized = True
    
    def search(
        self,
        query: str,
        outcome: Optional[str] = None,
        domain: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search for relevant turns.
        
        Args:
            query: Search query.
            outcome: Optional outcome filter.
            domain: Optional domain filter.
            top_k: Number of results.
            
        Returns:
            List of SearchResult objects.
        """
        if not self._initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        
        return self.vector_store.search(
            query=query,
            top_k=top_k or self.config.rag.top_k_retrieval,
            outcome_filter=outcome,
            domain_filter=domain,
            min_similarity=self.config.rag.similarity_threshold
        )
    
    def get_conversation(self, transcript_id: str) -> List[TurnDocument]:
        """Get full conversation by transcript ID."""
        return self.vector_store.get_conversation_turns(transcript_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        return self.vector_store.get_statistics()


# Utility functions for external use
def create_embedding_store(
    documents: List[TurnDocument],
    config=None,
    force_rebuild: bool = False
) -> EmbeddingStore:
    """
    Create and initialize an embedding store.
    
    Args:
        documents: List of TurnDocument objects.
        config: Optional configuration.
        force_rebuild: Force rebuilding index.
        
    Returns:
        Initialized EmbeddingStore.
    """
    store = EmbeddingStore(config)
    store.initialize(documents, force_rebuild)
    return store


if __name__ == "__main__":
    # Test the embedding store
    from data_processor import load_and_process
    
    processor, turns = load_and_process()
    
    print(f"\nLoaded {len(turns)} turn documents")
    
    # Create embedding store (use a subset for testing)
    test_turns = turns[:1000]
    store = create_embedding_store(test_turns, force_rebuild=True)
    
    print("\n=== Store Statistics ===")
    stats = store.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test search
    print("\n=== Test Search ===")
    results = store.search(
        query="customer frustrated supervisor escalation",
        outcome="Escalation",
        top_k=5
    )
    
    for result in results:
        print(f"\nScore: {result.score:.3f}")
        print(f"Transcript: {result.document.transcript_id}")
        print(f"Speaker: {result.document.speaker}")
        print(f"Text: {result.document.text[:100]}...")
