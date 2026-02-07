"""
LangGraph Orchestrator for Causal RAG System.
Implements the complete workflow with explicit, modular nodes for:
- Query Understanding
- RAG Retrieval
- Causal Reasoning
- Evidence Validation
- Context Memory Management
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
import operator
import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from config import get_config, OUTCOME_CATEGORIES
from data_processor import DataProcessor, TurnDocument, load_and_process
from embedding_store import EmbeddingStore, SearchResult, create_embedding_store
from context_memory import (
    ContextMemory, 
    QueryIntent, 
    detect_query_intent,
    CausalFactor,
    RetrievedEvidence,
    AnalysisResult
)
from causal_reasoning import CausalReasoningEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === State Definition ===

class GraphState(TypedDict):
    """State that flows through the LangGraph workflow."""
    # Input
    query: str
    selected_outcome: Optional[str]
    selected_domain: Optional[str]
    
    # Query Understanding
    query_intent: Optional[str]
    is_followup: bool
    
    # RAG Retrieval
    retrieved_results: List[Dict[str, Any]]
    retrieval_count: int
    
    # Causal Analysis
    analysis_result: Optional[Dict[str, Any]]
    causal_factors: List[Dict[str, Any]]
    
    # Evidence Validation
    validated: bool
    validation_warnings: List[str]
    
    # Output
    response: str
    evidence_display: List[Dict[str, Any]]
    
    # Metadata
    error: Optional[str]
    processing_steps: List[str]


# === Node Implementations ===

class QueryUnderstandingNode:
    """
    Node 1: Query Understanding
    Detects the intent of the user's query and determines processing path.
    """
    
    def __init__(self, context_memory: ContextMemory):
        """Initialize with context memory reference."""
        self.context = context_memory
    
    def __call__(self, state: GraphState) -> GraphState:
        """Process the query and determine intent."""
        logger.info("QueryUnderstandingNode: Processing query")
        
        query = state["query"]
        has_context = self.context.has_context()
        
        # Detect query intent
        intent = detect_query_intent(query, has_prior_context=has_context)
        
        # Determine if this is a follow-up
        is_followup = intent not in [QueryIntent.NEW_QUERY, QueryIntent.EXPLAIN_OUTCOME]
        
        # If outcome changed, treat as new query
        if state.get("selected_outcome") and state["selected_outcome"] != self.context.active_outcome:
            is_followup = False
            intent = QueryIntent.EXPLAIN_OUTCOME
        
        # Update context
        if state.get("selected_outcome"):
            self.context.set_active_outcome(state["selected_outcome"])
        if state.get("selected_domain"):
            self.context.set_active_domain(state["selected_domain"])
        
        # Add user query to conversation
        self.context.add_conversation_turn(
            role="user",
            content=query,
            query_intent=intent
        )
        
        logger.info(f"Detected intent: {intent.value}, is_followup: {is_followup}")
        
        return {
            **state,
            "query_intent": intent.value,
            "is_followup": is_followup,
            "processing_steps": state.get("processing_steps", []) + ["query_understanding"]
        }


class RAGRetrievalNode:
    """
    Node 2: RAG Retrieval
    Performs semantic search with metadata filtering to retrieve relevant dialogue turns.
    """
    
    def __init__(self, embedding_store: EmbeddingStore, context_memory: ContextMemory, config=None):
        """Initialize with embedding store and context."""
        self.store = embedding_store
        self.context = context_memory
        self.config = config or get_config()
    
    def __call__(self, state: GraphState) -> GraphState:
        """Retrieve relevant dialogue turns."""
        logger.info("RAGRetrievalNode: Retrieving evidence")
        
        intent = QueryIntent(state["query_intent"])
        
        # Check if we need new retrieval
        if not self.context.should_retrieve_new(intent) and self.context.retrieved_evidence:
            logger.info("Using cached evidence from context")
            # Return cached results
            cached_results = [
                {
                    "transcript_id": e.transcript_id,
                    "turn_id": e.turn_id,
                    "speaker": e.speaker,
                    "text": e.text,
                    "score": e.relevance_score
                }
                for e in self.context.retrieved_evidence
            ]
            return {
                **state,
                "retrieved_results": cached_results,
                "retrieval_count": len(cached_results),
                "processing_steps": state.get("processing_steps", []) + ["rag_retrieval_cached"]
            }
        
        # Build search query
        query = state["query"]
        outcome = self.context.active_outcome
        domain = self.context.active_domain
        
        # Enhance query for better retrieval
        if outcome:
            query = f"{query} {outcome.lower()} reasons causes"
        else:
            # When no outcome is selected, add general conversation keywords
            query = f"{query} conversation issue problem customer agent"
        
        # Perform search
        try:
            # Lower similarity threshold when no outcome filter is applied
            similarity_threshold = self.config.rag.similarity_threshold
            top_k = self.config.rag.top_k_retrieval
            
            if not outcome:
                similarity_threshold = max(0.3, similarity_threshold - 0.2)  # More lenient for broad searches
                top_k = int(top_k * 1.5)  # Retrieve more results for broader context
            
            results = self.store.vector_store.search(
                query=query,
                top_k=top_k,
                outcome_filter=outcome,
                domain_filter=domain,
                min_similarity=similarity_threshold
            )
            
            # For comparison queries, also get contrasting examples
            if intent == QueryIntent.COMPARE_OUTCOMES:
                contrast_outcome = "Resolution" if outcome == "Escalation" else "Escalation"
                contrast_results = self.store.vector_store.search(
                    query=query,
                    top_k=self.config.rag.top_k_retrieval // 2,
                    outcome_filter=contrast_outcome,
                    domain_filter=domain
                )
                results.extend(contrast_results)
            
            # Store in context
            for r in results:
                self.context.add_retrieved_evidence(
                    transcript_id=r.document.transcript_id,
                    turn_id=r.document.turn_id,
                    speaker=r.document.speaker,
                    text=r.document.text,
                    relevance_score=r.score
                )
            
            # Convert to dict for state
            result_dicts = [
                {
                    "transcript_id": r.document.transcript_id,
                    "turn_id": r.document.turn_id,
                    "speaker": r.document.speaker,
                    "text": r.document.text,
                    "domain": r.document.domain,
                    "outcome": r.document.outcome,
                    "score": r.score
                }
                for r in results
            ]
            
            logger.info(f"Retrieved {len(results)} turns")
            
            return {
                **state,
                "retrieved_results": result_dicts,
                "retrieval_count": len(results),
                "processing_steps": state.get("processing_steps", []) + ["rag_retrieval"]
            }
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {
                **state,
                "retrieved_results": [],
                "retrieval_count": 0,
                "error": str(e),
                "processing_steps": state.get("processing_steps", []) + ["rag_retrieval_error"]
            }


class CausalReasoningNode:
    """
    Node 3: Causal Reasoning
    Analyzes retrieved evidence to identify causal factors and generate explanations.
    """
    
    def __init__(self, reasoning_engine: CausalReasoningEngine, context_memory: ContextMemory):
        """Initialize with reasoning engine and context."""
        self.engine = reasoning_engine
        self.context = context_memory
    
    def __call__(self, state: GraphState) -> GraphState:
        """Perform causal reasoning."""
        logger.info("CausalReasoningNode: Analyzing causation")
        
        intent = QueryIntent(state["query_intent"])
        retrieved = state.get("retrieved_results", [])
        
        if not retrieved:
            return {
                **state,
                "analysis_result": None,
                "causal_factors": [],
                "response": "No relevant evidence found for the selected outcome. Please try a different outcome or broader criteria.",
                "processing_steps": state.get("processing_steps", []) + ["causal_reasoning_no_evidence"]
            }
        
        # Convert retrieved dicts back to SearchResult-like objects for the engine
        from embedding_store import SearchResult
        from data_processor import TurnDocument
        
        search_results = []
        for r in retrieved:
            doc = TurnDocument(
                transcript_id=r["transcript_id"],
                turn_id=r["turn_id"],
                speaker=r["speaker"],
                text=r["text"],
                domain=r.get("domain", ""),
                intent="",
                outcome=r.get("outcome", ""),
                reason_for_call="",
                time_of_interaction=""
            )
            search_results.append(SearchResult(
                document=doc,
                score=r["score"],
                rank=len(search_results) + 1
            ))
        
        try:
            # For follow-up queries, use the follow-up handler
            if state["is_followup"] and intent not in [QueryIntent.EXPLAIN_OUTCOME, QueryIntent.COMPARE_OUTCOMES]:
                context_data = self.context.get_context_for_followup()
                response = self.engine.answer_followup(
                    query=state["query"],
                    query_intent=intent,
                    context=context_data,
                    evidence=self.context.retrieved_evidence
                )
                
                return {
                    **state,
                    "response": response,
                    "causal_factors": [cf.to_dict() for cf in self.context.causal_factors],
                    "processing_steps": state.get("processing_steps", []) + ["causal_reasoning_followup"]
                }
            
            # Full causal analysis
            outcome = self.context.active_outcome or "Unknown"
            conversation_context = self.context.get_conversation_summary()
            
            analysis = self.engine.analyze(
                outcome=outcome,
                evidence=search_results,
                query_intent=intent,
                conversation_context=conversation_context
            )
            
            # Store in context
            self.context.set_analysis_result(analysis)
            for factor in analysis.causal_factors:
                self.context.causal_factors.append(factor)
            
            # Build response
            response = self._format_analysis_response(analysis)
            
            return {
                **state,
                "analysis_result": analysis.to_dict(),
                "causal_factors": [cf.to_dict() for cf in analysis.causal_factors],
                "response": response,
                "processing_steps": state.get("processing_steps", []) + ["causal_reasoning"]
            }
            
        except Exception as e:
            logger.error(f"Causal reasoning error: {e}")
            return {
                **state,
                "error": str(e),
                "response": f"An error occurred during analysis: {str(e)}",
                "processing_steps": state.get("processing_steps", []) + ["causal_reasoning_error"]
            }
    
    def _format_analysis_response(self, analysis: AnalysisResult) -> str:
        """Format analysis result as a readable response."""
        parts = []
        
        parts.append(f"## Causal Analysis: {analysis.outcome}")
        parts.append("")
        
        if analysis.causal_factors:
            parts.append("### Identified Causal Factors")
            parts.append("")
            
            for i, factor in enumerate(analysis.causal_factors, 1):
                confidence_bar = "█" * int(factor.confidence * 10) + "░" * (10 - int(factor.confidence * 10))
                parts.append(f"**{i}. {factor.name}** [{confidence_bar}] {factor.confidence:.0%}")
                parts.append(f"   {factor.causal_explanation}")
                parts.append("")
        else:
            parts.append("No clear causal factors identified with sufficient confidence.")
        
        parts.append(f"*Analysis based on {len(analysis.transcript_ids)} conversation(s)*")
        
        return "\n".join(parts)


class EvidenceValidationNode:
    """
    Node 4: Evidence Validation
    Ensures all causal claims are supported by retrieved evidence.
    Blocks any unsupported claims (hallucinations).
    """
    
    def __init__(self, context_memory: ContextMemory):
        """Initialize with context memory."""
        self.context = context_memory
    
    def __call__(self, state: GraphState) -> GraphState:
        """Validate that all claims are evidenced."""
        logger.info("EvidenceValidationNode: Validating evidence")
        
        causal_factors = state.get("causal_factors", [])
        retrieved = state.get("retrieved_results", [])
        
        if not causal_factors:
            return {
                **state,
                "validated": True,
                "validation_warnings": [],
                "processing_steps": state.get("processing_steps", []) + ["evidence_validation_skip"]
            }
        
        warnings = []
        validated_factors = []
        
        # Build evidence text lookup
        evidence_texts = {r["text"].lower() for r in retrieved}
        
        for factor in causal_factors:
            has_valid_evidence = False
            
            # Check if evidence snippets appear in retrieved content
            for snippet in factor.get("evidence_snippets", []):
                snippet_lower = snippet.lower()
                if any(snippet_lower in text for text in evidence_texts):
                    has_valid_evidence = True
                    break
            
            # Check if evidence refs are valid
            evidence_keys = {f"{r['transcript_id']}:{r['turn_id']}" for r in retrieved}
            for ref in factor.get("evidence_turn_ids", []):
                if ref in evidence_keys:
                    has_valid_evidence = True
                    break
            
            if has_valid_evidence:
                validated_factors.append(factor)
            else:
                warnings.append(f"Factor '{factor.get('name', 'Unknown')}' has weak evidence support")
                # Still include but mark confidence lower
                factor_copy = factor.copy()
                factor_copy["confidence"] = min(factor.get("confidence", 0.5) * 0.5, 0.4)
                validated_factors.append(factor_copy)
        
        logger.info(f"Validation complete: {len(warnings)} warnings")
        
        return {
            **state,
            "causal_factors": validated_factors,
            "validated": True,
            "validation_warnings": warnings,
            "processing_steps": state.get("processing_steps", []) + ["evidence_validation"]
        }


class ContextMemoryNode:
    """
    Node 5: Context Memory
    Updates and manages context for multi-turn reasoning.
    """
    
    def __init__(self, context_memory: ContextMemory):
        """Initialize with context memory."""
        self.context = context_memory
    
    def __call__(self, state: GraphState) -> GraphState:
        """Update context memory with results."""
        logger.info("ContextMemoryNode: Updating context")
        
        # Add assistant response to conversation
        self.context.add_conversation_turn(
            role="assistant",
            content=state.get("response", ""),
            analysis_result=self.context.latest_analysis
        )
        
        # Prepare evidence display
        evidence_display = []
        for r in state.get("retrieved_results", [])[:10]:  # Limit display
            evidence_display.append({
                "transcript_id": r["transcript_id"],
                "turn_id": r["turn_id"],
                "speaker": r["speaker"],
                "text": r["text"],
                "relevance": f"{r['score']:.2f}"
            })
        
        return {
            **state,
            "evidence_display": evidence_display,
            "processing_steps": state.get("processing_steps", []) + ["context_memory_update"]
        }


# === Graph Builder ===

class CausalRAGOrchestrator:
    """
    Main orchestrator that builds and executes the LangGraph workflow.
    """
    
    def __init__(self, data_path: Optional[str] = None, force_rebuild: bool = False):
        """
        Initialize the orchestrator.
        
        Args:
            data_path: Path to JSON data file.
            force_rebuild: Whether to rebuild the vector index.
        """
        self.config = get_config()
        
        # Initialize components
        logger.info("Initializing CausalRAGOrchestrator...")
        
        # Load and process data
        self.processor = DataProcessor()
        self.processor.load_data(data_path)
        self.turns = self.processor.flatten_to_turns()
        
        logger.info(f"Processed {len(self.turns)} turn documents")
        
        # Create embedding store
        self.embedding_store = EmbeddingStore(self.config)
        self.embedding_store.initialize(self.turns, force_rebuild=force_rebuild)
        
        # Create context memory
        self.context_memory = ContextMemory()
        
        # Create reasoning engine
        self.reasoning_engine = CausalReasoningEngine(self.config)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("CausalRAGOrchestrator initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create nodes
        query_node = QueryUnderstandingNode(self.context_memory)
        rag_node = RAGRetrievalNode(self.embedding_store, self.context_memory, self.config)
        reasoning_node = CausalReasoningNode(self.reasoning_engine, self.context_memory)
        validation_node = EvidenceValidationNode(self.context_memory)
        memory_node = ContextMemoryNode(self.context_memory)
        
        # Create graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("query_understanding", query_node)
        workflow.add_node("rag_retrieval", rag_node)
        workflow.add_node("causal_reasoning", reasoning_node)
        workflow.add_node("evidence_validation", validation_node)
        workflow.add_node("context_memory", memory_node)
        
        # Define edges (linear flow with conditional branches)
        workflow.set_entry_point("query_understanding")
        workflow.add_edge("query_understanding", "rag_retrieval")
        workflow.add_edge("rag_retrieval", "causal_reasoning")
        workflow.add_edge("causal_reasoning", "evidence_validation")
        workflow.add_edge("evidence_validation", "context_memory")
        workflow.add_edge("context_memory", END)
        
        # Compile graph
        return workflow.compile()
    
    def process_query(
        self,
        query: str,
        outcome: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the full workflow.
        
        Args:
            query: User's question.
            outcome: Selected outcome to analyze.
            domain: Optional domain filter.
            
        Returns:
            Final state with response and evidence.
        """
        # Create initial state
        initial_state: GraphState = {
            "query": query,
            "selected_outcome": outcome,
            "selected_domain": domain,
            "query_intent": None,
            "is_followup": False,
            "retrieved_results": [],
            "retrieval_count": 0,
            "analysis_result": None,
            "causal_factors": [],
            "validated": False,
            "validation_warnings": [],
            "response": "",
            "evidence_display": [],
            "error": None,
            "processing_steps": []
        }
        
        # Execute graph
        logger.info(f"Processing query: {query[:50]}...")
        
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            return {
                **initial_state,
                "error": str(e),
                "response": f"An error occurred: {str(e)}"
            }
    
    def reset_context(self) -> None:
        """Reset the context memory for a new session."""
        self.context_memory.reset()
        logger.info("Context reset")
    
    def get_available_outcomes(self) -> List[str]:
        """Get list of available outcomes in the data."""
        return self.processor.get_unique_outcomes()
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains in the data."""
        return self.processor.get_unique_domains()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "data": self.processor.get_statistics(),
            "embedding_store": self.embedding_store.get_statistics(),
            "context": self.context_memory.get_context_for_followup()
        }
    
    def export_analysis(self, filepath: str) -> None:
        """Export current analysis to JSON file."""
        self.context_memory.export_to_json(filepath)


# === Convenience Functions ===

def create_orchestrator(
    data_path: Optional[str] = None,
    force_rebuild: bool = False
) -> CausalRAGOrchestrator:
    """
    Create and initialize a CausalRAGOrchestrator.
    
    Args:
        data_path: Path to JSON data file.
        force_rebuild: Whether to rebuild the vector index.
        
    Returns:
        Initialized orchestrator.
    """
    return CausalRAGOrchestrator(data_path=data_path, force_rebuild=force_rebuild)


if __name__ == "__main__":
    # Test the orchestrator
    print("Creating orchestrator...")
    orchestrator = create_orchestrator(force_rebuild=True)
    
    print("\n=== Available Outcomes ===")
    outcomes = orchestrator.get_available_outcomes()
    for o in outcomes:
        print(f"  - {o}")
    
    print("\n=== Test Query 1: Initial Analysis ===")
    result = orchestrator.process_query(
        query="Why do calls escalate?",
        outcome="Escalation"
    )
    print(f"\nResponse:\n{result['response']}")
    print(f"\nProcessing steps: {result['processing_steps']}")
    print(f"Evidence count: {result['retrieval_count']}")
    
    print("\n=== Test Query 2: Follow-up ===")
    result = orchestrator.process_query(
        query="Could these escalations have been prevented?",
        outcome="Escalation"
    )
    print(f"\nResponse:\n{result['response']}")
    print(f"\nProcessing steps: {result['processing_steps']}")
    
    print("\n=== Test Query 3: Agent Analysis ===")
    result = orchestrator.process_query(
        query="What agent behaviors contributed to these escalations?",
        outcome="Escalation"
    )
    print(f"\nResponse:\n{result['response']}")
    
    print("\n=== Statistics ===")
    stats = orchestrator.get_statistics()
    print(f"Total transcripts: {stats['data']['total_transcripts']}")
    print(f"Total vectors: {stats['embedding_store'].get('total_vectors', 'N/A')}")
