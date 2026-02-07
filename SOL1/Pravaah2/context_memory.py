"""
Context Memory Module for Causal RAG System.
Manages conversation state for multi-turn reasoning with deterministic context preservation.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib

from data_processor import TurnDocument

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of user query intents."""
    EXPLAIN_OUTCOME = "explain_outcome"  # Why did X happen?
    COMPARE_OUTCOMES = "compare_outcomes"  # Compare escalated vs resolved
    DRILL_DOWN = "drill_down"  # More details on a factor
    PREVENTION = "prevention"  # Could this have been prevented?
    AGENT_ANALYSIS = "agent_analysis"  # Which agent behaviors mattered?
    CUSTOMER_ANALYSIS = "customer_analysis"  # Customer behavior patterns
    SHOW_EVIDENCE = "show_evidence"  # Show me the evidence
    NEW_QUERY = "new_query"  # Starting fresh
    FOLLOW_UP = "follow_up"  # General follow-up


@dataclass
class CausalFactor:
    """Represents an identified causal factor."""
    factor_id: str
    name: str
    description: str
    causal_explanation: str
    confidence: float
    evidence_turn_ids: List[str]  # transcript_id:turn_id
    evidence_snippets: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RetrievedEvidence:
    """Represents retrieved evidence for a causal analysis."""
    transcript_id: str
    turn_id: int
    speaker: str
    text: str
    relevance_score: float
    
    def get_key(self) -> str:
        """Get unique key for this evidence."""
        return f"{self.transcript_id}:{self.turn_id}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AnalysisResult:
    """Represents the result of a causal analysis."""
    outcome: str
    causal_factors: List[CausalFactor]
    evidence: List[RetrievedEvidence]
    transcript_ids: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcome": self.outcome,
            "causal_factors": [cf.to_dict() for cf in self.causal_factors],
            "evidence": [e.to_dict() for e in self.evidence],
            "transcript_ids": self.transcript_ids,
            "timestamp": self.timestamp
        }


@dataclass
class ConversationTurn:
    """Represents a turn in the user-system conversation."""
    turn_id: int
    role: str  # "user" or "assistant"
    content: str
    query_intent: Optional[QueryIntent] = None
    analysis_result: Optional[AnalysisResult] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "query_intent": self.query_intent.value if self.query_intent else None,
            "analysis_result": self.analysis_result.to_dict() if self.analysis_result else None,
            "timestamp": self.timestamp
        }


class ContextMemory:
    """
    Manages conversation context for multi-turn causal reasoning.
    
    Key responsibilities:
    - Store active outcome and domain filters
    - Track retrieved transcript IDs and evidence
    - Preserve identified causal factors
    - Maintain conversation history
    - Enable context reuse for follow-up queries
    """
    
    def __init__(self):
        """Initialize context memory."""
        self.session_id: str = self._generate_session_id()
        
        # Active analysis state
        self.active_outcome: Optional[str] = None
        self.active_domain: Optional[str] = None
        
        # Retrieved context
        self.retrieved_transcript_ids: Set[str] = set()
        self.retrieved_evidence: List[RetrievedEvidence] = []
        
        # Causal analysis results
        self.causal_factors: List[CausalFactor] = []
        self.latest_analysis: Optional[AnalysisResult] = None
        
        # Conversation history
        self.conversation_history: List[ConversationTurn] = []
        self._turn_counter: int = 0
        
        # Logging
        self.retrieval_log: List[Dict[str, Any]] = []
        
        logger.info(f"ContextMemory initialized with session_id: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def set_active_outcome(self, outcome: str) -> None:
        """
        Set the active outcome for analysis.
        
        Args:
            outcome: Outcome category (e.g., "Escalation").
        """
        if self.active_outcome != outcome:
            logger.info(f"Active outcome changed: {self.active_outcome} -> {outcome}")
            self.active_outcome = outcome
            # Clear previous analysis when outcome changes
            self._clear_analysis_state()
    
    def set_active_domain(self, domain: Optional[str]) -> None:
        """
        Set the active domain filter.
        
        Args:
            domain: Domain to filter by (e.g., "Healthcare Services").
        """
        self.active_domain = domain
        logger.info(f"Active domain set: {domain}")
    
    def _clear_analysis_state(self) -> None:
        """Clear analysis state when starting fresh."""
        self.retrieved_transcript_ids.clear()
        self.retrieved_evidence.clear()
        self.causal_factors.clear()
        self.latest_analysis = None
        self.retrieval_log.clear()
        logger.info("Analysis state cleared")
    
    def reset(self) -> None:
        """
        Fully reset the context memory.
        """
        self.active_outcome = None
        self.active_domain = None
        self._clear_analysis_state()
        self.conversation_history.clear()
        self._turn_counter = 0
        self.session_id = self._generate_session_id()
        logger.info(f"Context memory reset. New session_id: {self.session_id}")
    
    def add_retrieved_evidence(
        self,
        transcript_id: str,
        turn_id: int,
        speaker: str,
        text: str,
        relevance_score: float
    ) -> RetrievedEvidence:
        """
        Add retrieved evidence to context.
        
        Args:
            transcript_id: Transcript ID.
            turn_id: Turn number in the transcript.
            speaker: Speaker (Agent/Customer).
            text: Turn text.
            relevance_score: Similarity score.
            
        Returns:
            Created RetrievedEvidence object.
        """
        evidence = RetrievedEvidence(
            transcript_id=transcript_id,
            turn_id=turn_id,
            speaker=speaker,
            text=text,
            relevance_score=relevance_score
        )
        
        # Avoid duplicates
        existing_keys = {e.get_key() for e in self.retrieved_evidence}
        if evidence.get_key() not in existing_keys:
            self.retrieved_evidence.append(evidence)
            self.retrieved_transcript_ids.add(transcript_id)
            
            # Log retrieval
            self.retrieval_log.append({
                "timestamp": datetime.now().isoformat(),
                "transcript_id": transcript_id,
                "turn_id": turn_id,
                "score": relevance_score
            })
        
        return evidence
    
    def add_causal_factor(
        self,
        name: str,
        description: str,
        causal_explanation: str,
        confidence: float,
        evidence_turn_ids: List[str],
        evidence_snippets: List[str]
    ) -> CausalFactor:
        """
        Add an identified causal factor.
        
        Args:
            name: Short name for the factor.
            description: Longer description.
            causal_explanation: Causal chain explanation.
            confidence: Confidence score (0-1).
            evidence_turn_ids: List of transcript_id:turn_id references.
            evidence_snippets: Actual text snippets as evidence.
            
        Returns:
            Created CausalFactor object.
        """
        factor_id = f"cf_{len(self.causal_factors) + 1}"
        
        factor = CausalFactor(
            factor_id=factor_id,
            name=name,
            description=description,
            causal_explanation=causal_explanation,
            confidence=confidence,
            evidence_turn_ids=evidence_turn_ids,
            evidence_snippets=evidence_snippets
        )
        
        self.causal_factors.append(factor)
        logger.info(f"Added causal factor: {name} (confidence: {confidence:.2f})")
        
        return factor
    
    def set_analysis_result(self, result: AnalysisResult) -> None:
        """
        Store the latest analysis result.
        
        Args:
            result: AnalysisResult object.
        """
        self.latest_analysis = result
        logger.info(f"Analysis result stored for outcome: {result.outcome}")
    
    def add_conversation_turn(
        self,
        role: str,
        content: str,
        query_intent: Optional[QueryIntent] = None,
        analysis_result: Optional[AnalysisResult] = None
    ) -> ConversationTurn:
        """
        Add a turn to conversation history.
        
        Args:
            role: "user" or "assistant".
            content: Turn content.
            query_intent: Detected intent (for user turns).
            analysis_result: Analysis result (for assistant turns).
            
        Returns:
            Created ConversationTurn object.
        """
        self._turn_counter += 1
        
        turn = ConversationTurn(
            turn_id=self._turn_counter,
            role=role,
            content=content,
            query_intent=query_intent,
            analysis_result=analysis_result
        )
        
        self.conversation_history.append(turn)
        
        return turn
    
    def get_context_for_followup(self) -> Dict[str, Any]:
        """
        Get context needed for processing follow-up queries.
        
        Returns:
            Dictionary with active context state.
        """
        return {
            "session_id": self.session_id,
            "active_outcome": self.active_outcome,
            "active_domain": self.active_domain,
            "retrieved_transcript_ids": list(self.retrieved_transcript_ids),
            "causal_factors": [cf.to_dict() for cf in self.causal_factors],
            "evidence_count": len(self.retrieved_evidence),
            "conversation_turns": len(self.conversation_history),
            "has_analysis": self.latest_analysis is not None
        }
    
    def get_evidence_for_factor(self, factor_id: str) -> List[RetrievedEvidence]:
        """
        Get evidence associated with a specific causal factor.
        
        Args:
            factor_id: Factor ID to look up.
            
        Returns:
            List of evidence items.
        """
        # Find the factor
        factor = next((cf for cf in self.causal_factors if cf.factor_id == factor_id), None)
        if not factor:
            return []
        
        # Get matching evidence
        evidence_keys = set(factor.evidence_turn_ids)
        return [e for e in self.retrieved_evidence if e.get_key() in evidence_keys]
    
    def get_recent_conversation(self, n_turns: int = 5) -> List[ConversationTurn]:
        """
        Get the most recent conversation turns.
        
        Args:
            n_turns: Number of turns to return.
            
        Returns:
            List of recent ConversationTurn objects.
        """
        return self.conversation_history[-n_turns:] if self.conversation_history else []
    
    def should_retrieve_new(self, query_intent: QueryIntent) -> bool:
        """
        Determine if new retrieval is needed based on query intent.
        
        Args:
            query_intent: Detected query intent.
            
        Returns:
            True if new retrieval is needed.
        """
        # Always retrieve for new queries or comparisons
        if query_intent in [QueryIntent.NEW_QUERY, QueryIntent.COMPARE_OUTCOMES]:
            return True
        
        # Don't retrieve for drill-down or show evidence if we have context
        if query_intent in [QueryIntent.DRILL_DOWN, QueryIntent.SHOW_EVIDENCE]:
            return not self.has_context()
        
        # For other follow-ups, retrieve if no context
        return not self.has_context()
    
    def has_context(self) -> bool:
        """Check if we have active analysis context."""
        return (
            self.active_outcome is not None and 
            len(self.retrieved_evidence) > 0
        )
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation for LLM context.
        
        Returns:
            Summary string.
        """
        if not self.conversation_history:
            return "No prior conversation."
        
        summary_parts = []
        
        # Include active context
        if self.active_outcome:
            summary_parts.append(f"Analyzing outcome: {self.active_outcome}")
        
        if self.causal_factors:
            factor_names = [cf.name for cf in self.causal_factors]
            summary_parts.append(f"Identified factors: {', '.join(factor_names)}")
        
        # Include recent turns
        recent = self.get_recent_conversation(3)
        if recent:
            summary_parts.append("Recent conversation:")
            for turn in recent:
                role = "User" if turn.role == "user" else "Assistant"
                content_preview = turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
                summary_parts.append(f"  {role}: {content_preview}")
        
        return "\n".join(summary_parts)
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export full context state for persistence.
        
        Returns:
            Dictionary with full state.
        """
        return {
            "session_id": self.session_id,
            "active_outcome": self.active_outcome,
            "active_domain": self.active_domain,
            "retrieved_transcript_ids": list(self.retrieved_transcript_ids),
            "retrieved_evidence": [e.to_dict() for e in self.retrieved_evidence],
            "causal_factors": [cf.to_dict() for cf in self.causal_factors],
            "latest_analysis": self.latest_analysis.to_dict() if self.latest_analysis else None,
            "conversation_history": [t.to_dict() for t in self.conversation_history],
            "retrieval_log": self.retrieval_log
        }
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export state to JSON file.
        
        Args:
            filepath: Path to output file.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export_state(), f, indent=2, ensure_ascii=False)
        logger.info(f"Context exported to {filepath}")


def detect_query_intent(query: str, has_prior_context: bool = False) -> QueryIntent:
    """
    Detect the intent of a user query.
    
    Args:
        query: User's query text.
        has_prior_context: Whether there is existing context.
        
    Returns:
        Detected QueryIntent.
    """
    query_lower = query.lower()
    
    # Check for specific patterns
    if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'non-escalated']):
        return QueryIntent.COMPARE_OUTCOMES
    
    if any(word in query_lower for word in ['prevent', 'avoided', 'could have', 'would have']):
        return QueryIntent.PREVENTION
    
    if any(word in query_lower for word in ['agent', 'representative', 'support staff']):
        return QueryIntent.AGENT_ANALYSIS
    
    if any(word in query_lower for word in ['customer', 'caller', 'client']):
        return QueryIntent.CUSTOMER_ANALYSIS
    
    if any(word in query_lower for word in ['show', 'evidence', 'proof', 'example', 'quote']):
        return QueryIntent.SHOW_EVIDENCE
    
    if any(word in query_lower for word in ['more', 'detail', 'elaborate', 'explain further']):
        return QueryIntent.DRILL_DOWN
    
    if any(word in query_lower for word in ['why', 'cause', 'reason', 'what led']):
        if has_prior_context:
            return QueryIntent.FOLLOW_UP
        return QueryIntent.EXPLAIN_OUTCOME
    
    # Default based on context
    if has_prior_context:
        return QueryIntent.FOLLOW_UP
    return QueryIntent.NEW_QUERY


if __name__ == "__main__":
    # Test the context memory
    memory = ContextMemory()
    
    # Simulate an analysis session
    memory.set_active_outcome("Escalation")
    
    # Add some evidence
    memory.add_retrieved_evidence(
        transcript_id="123-456",
        turn_id=5,
        speaker="Customer",
        text="I've been trying to resolve this for three weeks!",
        relevance_score=0.85
    )
    
    # Add a causal factor
    memory.add_causal_factor(
        name="Repeated Unresolved Issues",
        description="Customer mentioned trying multiple times without resolution",
        causal_explanation="Repeated failures led to frustration and escalation request",
        confidence=0.9,
        evidence_turn_ids=["123-456:5"],
        evidence_snippets=["I've been trying to resolve this for three weeks!"]
    )
    
    # Add conversation turn
    memory.add_conversation_turn(
        role="user",
        content="Why did this call escalate?",
        query_intent=QueryIntent.EXPLAIN_OUTCOME
    )
    
    print("=== Context State ===")
    context = memory.get_context_for_followup()
    for key, value in context.items():
        print(f"{key}: {value}")
    
    print("\n=== Conversation Summary ===")
    print(memory.get_conversation_summary())
    
    # Test intent detection
    print("\n=== Intent Detection ===")
    test_queries = [
        "Why did this escalate?",
        "Could this have been prevented?",
        "Compare with non-escalated calls",
        "Show me the evidence",
        "What about the agent's behavior?"
    ]
    
    for q in test_queries:
        intent = detect_query_intent(q, has_prior_context=True)
        print(f"'{q}' -> {intent.value}")
