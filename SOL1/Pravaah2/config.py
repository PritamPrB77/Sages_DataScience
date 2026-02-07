"""
Configuration module for Causal RAG Conversational Analysis System.
Contains all settings, model configurations, and system constants.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class EmbeddingModel(Enum):
    """Supported embedding models."""
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI = "text-embedding-3-large"


class VectorStore(Enum):
    """Supported vector stores."""
    FAISS = "faiss"
    CHROMA = "chromadb"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = EmbeddingModel.MINILM.value
    use_openai: bool = False
    openai_api_key: Optional[str] = None
    batch_size: int = 32
    normalize_embeddings: bool = True
    
    def __post_init__(self):
        if self.use_openai:
            self.model_name = EmbeddingModel.OPENAI.value
            self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    store_type: str = VectorStore.FAISS.value
    index_path: str = "./vector_index"
    similarity_metric: str = "cosine"  # cosine, l2, ip
    top_k: int = 10


@dataclass
class LLMConfig:
    """LLM configuration for reasoning (Gemini)."""
    model_name: str = "gemini-1.5-flash"  # Default to Gemini 1.5 Flash for cost efficiency
    api_key: Optional[str] = None
    temperature: float = 0.1  # Low temperature for deterministic reasoning
    max_tokens: int = 2000
    timeout: int = 60
    
    def __post_init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", self.api_key)


@dataclass
class TransformerConfig:
    """Transformer models for preprocessing/signals."""
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    dialogue_act_model: str = "roberta-base"  # Fine-tuned for dialogue acts
    use_gpu: bool = False
    batch_size: int = 16


@dataclass
class RAGConfig:
    """RAG retrieval configuration."""
    top_k_retrieval: int = 15
    similarity_threshold: float = 0.5
    rerank_enabled: bool = True
    max_context_turns: int = 50
    chunk_overlap: int = 2  # Overlapping turns for context


@dataclass
class CausalConfig:
    """Causal reasoning configuration."""
    min_evidence_count: int = 2
    confidence_threshold: float = 0.6
    max_causal_factors: int = 5
    enable_comparison: bool = True
    comparison_sample_size: int = 5


# Outcome categories for classification
OUTCOME_CATEGORIES = [
    "Escalation",
    "Complaint",
    "Refund",
    "Resolution",
    "Fraud Handled",
    "Service Recovery",
    "Customer Churn Risk",
    "Successful Resolution",
    "Unresolved"
]

# Intent to outcome mapping based on dataset analysis
INTENT_OUTCOME_MAPPING = {
    # Escalation outcomes
    "Escalation - Repeated Service Failures": "Escalation",
    "Escalation - Medical Error Complaint": "Escalation",
    "Escalation - Service Cancellation Threat": "Escalation",
    "Escalation - Threat of Legal Action": "Escalation",
    "Escalation - Unauthorized Account Closure": "Escalation",
    
    # Fraud-related outcomes
    "Fraud Alert Investigation": "Fraud Handled",
    "Multiple Issues - Fraud & Account Updates": "Fraud Handled",
    "Multiple Issues - Fraud, Account & Security": "Fraud Handled",
    
    # Service issues
    "Service Interruptions": "Service Recovery",
    "Update Failures": "Unresolved",
    "Account Access Issues": "Resolution",
    
    # Business events
    "Business Event - Cyber Attack": "Service Recovery",
    "Business Event - Data Breach Response": "Service Recovery",
    "Business Event - Network Outage": "Service Recovery",
    "Business Event - System Outage": "Service Recovery",
    "Business Event - Ransomware Attack": "Service Recovery",
    "Business Event - System Conversion Failure": "Unresolved",
    "Business Event - Product Recall": "Complaint",
    "Business Event - Warehouse Fire": "Service Recovery",
    "Business Event - Major Policy Changes": "Resolution",
    "Business Event - Policy Changes": "Resolution",
    
    # Resolution outcomes
    "Delivery Investigation": "Resolution",
    "Claim Denials": "Complaint",
    "Reservation Modifications": "Resolution",
    "Appointment Scheduling": "Resolution",
    
    # Multiple issues - typically resolved
    "Multiple Issues - Appointment, Prescription & Insurance": "Resolution",
    "Multiple Issues - Billing & Payment Setup": "Resolution",
    "Multiple Issues - Billing, Plan Changes & Equipment": "Resolution",
    "Multiple Issues - Claim, Coverage & Policy": "Resolution",
    "Multiple Issues - Claims, Coverage & Policy Updates": "Resolution",
    "Multiple Issues - Medical Records & Billing": "Resolution",
    "Multiple Issues - Order Status & Account Access": "Resolution",
    "Multiple Issues - Order Status, Billing & Account": "Resolution",
    "Multiple Issues - Payments & Policy Management": "Resolution",
    "Multiple Issues - Reservation, Service & Amenities": "Resolution",
    "Multiple Issues - Returns & Account Inquiries": "Resolution",
    "Multiple Issues - Scheduling, Prescriptions & Insurance": "Resolution",
    "Multiple Issues - Service & Billing Setup": "Resolution",
    "Multiple Issues - Service Complaints & Reservations": "Complaint",
    "Multiple Issues - Technical Support & Account Management": "Resolution",
    "Multiple Issues - Technical, Plan & Payment": "Resolution",
}

# Causal behavior patterns to detect
CAUSAL_BEHAVIOR_PATTERNS = {
    "Escalation": [
        "repeated_unresolved_issues",
        "agent_dismissiveness",
        "long_wait_times",
        "transfer_fatigue",
        "policy_rigidity",
        "lack_of_empathy",
        "communication_breakdown",
        "unfulfilled_promises"
    ],
    "Complaint": [
        "service_failure",
        "billing_errors",
        "poor_communication",
        "unmet_expectations",
        "quality_issues",
        "delayed_response"
    ],
    "Fraud Handled": [
        "quick_detection",
        "proactive_monitoring",
        "efficient_resolution",
        "customer_notification",
        "account_protection"
    ],
    "Resolution": [
        "active_listening",
        "problem_identification",
        "solution_offering",
        "follow_up_confirmation",
        "customer_satisfaction"
    ],
    "Refund": [
        "product_defect",
        "service_not_rendered",
        "billing_dispute",
        "policy_exception",
        "customer_dissatisfaction"
    ]
}

# Sentiment indicators for causal analysis
NEGATIVE_SENTIMENT_KEYWORDS = [
    "frustrated", "angry", "disappointed", "unacceptable", "ridiculous",
    "terrible", "awful", "worst", "never", "can't believe", "waste",
    "incompetent", "useless", "fed up", "done with", "cancel", "complaint",
    "supervisor", "manager", "escalate", "legal", "lawsuit", "lawyer"
]

POSITIVE_SENTIMENT_KEYWORDS = [
    "thank you", "appreciate", "helpful", "resolved", "satisfied",
    "excellent", "great", "wonderful", "perfect", "amazing", "impressed",
    "grateful", "happy", "pleased", "relief"
]

# Agent behavior indicators
AGENT_POSITIVE_BEHAVIORS = [
    "empathy", "apology", "ownership", "solution", "proactive",
    "verification", "confirmation", "follow-up", "personalization"
]

AGENT_NEGATIVE_BEHAVIORS = [
    "dismissive", "defensive", "policy_hiding", "blame_shifting",
    "lack_of_ownership", "scripted_response", "interruption"
]


@dataclass
class SystemConfig:
    """Main system configuration aggregating all configs."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    
    # Paths
    data_path: str = "./Conversational_Transcript_Dataset.json"
    log_path: str = "./logs"
    export_path: str = "./exports"
    
    # System settings
    debug_mode: bool = False
    log_retrieved_chunks: bool = True
    enable_caching: bool = True


# Global configuration instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> SystemConfig:
    """Update configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
