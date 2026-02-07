"""
Data Processor Module for Causal RAG System.
Handles loading, parsing, and transforming conversation transcripts into turn-level documents.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

from config import (
    get_config, 
    INTENT_OUTCOME_MAPPING, 
    NEGATIVE_SENTIMENT_KEYWORDS,
    POSITIVE_SENTIMENT_KEYWORDS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TurnDocument:
    """Represents a single turn in a conversation as a document for indexing."""
    transcript_id: str
    turn_id: int
    speaker: str
    text: str
    domain: str
    intent: str
    outcome: str
    reason_for_call: str
    time_of_interaction: str
    
    # Derived features
    sentiment_score: float = 0.0
    contains_escalation_signal: bool = False
    contains_complaint_signal: bool = False
    word_count: int = 0
    turn_position: str = "middle"  # start, middle, end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for vector store filtering."""
        return {
            "transcript_id": self.transcript_id,
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "domain": self.domain,
            "intent": self.intent,
            "outcome": self.outcome,
            "sentiment_score": self.sentiment_score,
            "contains_escalation_signal": self.contains_escalation_signal,
            "contains_complaint_signal": self.contains_complaint_signal,
            "turn_position": self.turn_position
        }


@dataclass 
class Transcript:
    """Represents a complete conversation transcript."""
    transcript_id: str
    time_of_interaction: str
    domain: str
    intent: str
    reason_for_call: str
    conversation: List[Dict[str, str]]
    outcome: str = ""
    
    def __post_init__(self):
        """Derive outcome from intent if not explicitly set."""
        if not self.outcome:
            self.outcome = INTENT_OUTCOME_MAPPING.get(self.intent, "Unknown")


class DataProcessor:
    """
    Processes conversation transcript data for the Causal RAG system.
    
    Responsibilities:
    - Load JSON data from disk
    - Parse and validate transcripts
    - Flatten conversations into turn-level documents
    - Compute derived features (sentiment, signals)
    - Preserve original wording
    """
    
    def __init__(self, config=None):
        """Initialize the data processor."""
        self.config = config or get_config()
        self.transcripts: List[Transcript] = []
        self.turn_documents: List[TurnDocument] = []
        self._loaded = False
        
    def load_data(self, file_path: Optional[str] = None) -> List[Transcript]:
        """
        Load conversation transcripts from JSON file.
        
        Args:
            file_path: Path to JSON file. Uses config default if not provided.
            
        Returns:
            List of Transcript objects.
        """
        path = Path(file_path or self.config.data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        logger.info(f"Loading data from {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both array and object with 'transcripts' key
        if isinstance(data, list):
            raw_transcripts = data
        elif isinstance(data, dict) and 'transcripts' in data:
            raw_transcripts = data['transcripts']
        else:
            raise ValueError("Invalid data format. Expected list or object with 'transcripts' key.")
        
        self.transcripts = []
        for raw in raw_transcripts:
            transcript = self._parse_transcript(raw)
            if transcript:
                self.transcripts.append(transcript)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.transcripts)} transcripts")
        
        return self.transcripts
    
    def _parse_transcript(self, raw: Dict[str, Any]) -> Optional[Transcript]:
        """
        Parse a raw transcript dictionary into a Transcript object.
        
        Args:
            raw: Raw transcript dictionary from JSON.
            
        Returns:
            Transcript object or None if invalid.
        """
        try:
            return Transcript(
                transcript_id=raw.get('transcript_id', ''),
                time_of_interaction=raw.get('time_of_interaction', ''),
                domain=raw.get('domain', ''),
                intent=raw.get('intent', ''),
                reason_for_call=raw.get('reason_for_call', ''),
                conversation=raw.get('conversation', [])
            )
        except Exception as e:
            logger.warning(f"Failed to parse transcript: {e}")
            return None
    
    def flatten_to_turns(self) -> List[TurnDocument]:
        """
        Flatten all transcripts into turn-level documents.
        
        Each turn becomes a separate document with full metadata,
        suitable for embedding and vector indexing.
        
        Returns:
            List of TurnDocument objects.
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        logger.info("Flattening transcripts to turn-level documents")
        
        self.turn_documents = []
        
        for transcript in self.transcripts:
            turns = self._process_transcript_turns(transcript)
            self.turn_documents.extend(turns)
        
        logger.info(f"Created {len(self.turn_documents)} turn documents")
        
        return self.turn_documents
    
    def _process_transcript_turns(self, transcript: Transcript) -> List[TurnDocument]:
        """
        Process all turns in a single transcript.
        
        Args:
            transcript: Transcript object to process.
            
        Returns:
            List of TurnDocument objects for this transcript.
        """
        turns = []
        conversation = transcript.conversation
        total_turns = len(conversation)
        
        for i, turn in enumerate(conversation):
            # Determine turn position
            if i == 0:
                position = "start"
            elif i == total_turns - 1:
                position = "end"
            else:
                position = "middle"
            
            # Compute sentiment and signals
            text = turn.get('text', '')
            sentiment = self._compute_simple_sentiment(text)
            escalation_signal = self._detect_escalation_signal(text)
            complaint_signal = self._detect_complaint_signal(text)
            
            turn_doc = TurnDocument(
                transcript_id=transcript.transcript_id,
                turn_id=i,
                speaker=turn.get('speaker', ''),
                text=text,  # Preserve original wording
                domain=transcript.domain,
                intent=transcript.intent,
                outcome=transcript.outcome,
                reason_for_call=transcript.reason_for_call,
                time_of_interaction=transcript.time_of_interaction,
                sentiment_score=sentiment,
                contains_escalation_signal=escalation_signal,
                contains_complaint_signal=complaint_signal,
                word_count=len(text.split()),
                turn_position=position
            )
            
            turns.append(turn_doc)
        
        return turns
    
    def _compute_simple_sentiment(self, text: str) -> float:
        """
        Compute a simple sentiment score based on keyword matching.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Sentiment score from -1 (negative) to 1 (positive).
        """
        text_lower = text.lower()
        
        negative_count = sum(1 for kw in NEGATIVE_SENTIMENT_KEYWORDS if kw in text_lower)
        positive_count = sum(1 for kw in POSITIVE_SENTIMENT_KEYWORDS if kw in text_lower)
        
        total = negative_count + positive_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def _detect_escalation_signal(self, text: str) -> bool:
        """
        Detect if text contains escalation signals.
        
        Args:
            text: Text to analyze.
            
        Returns:
            True if escalation signals detected.
        """
        escalation_keywords = [
            'supervisor', 'manager', 'escalate', 'speak to someone else',
            'higher authority', 'not acceptable', 'formal complaint',
            'legal action', 'lawyer', 'lawsuit', 'attorney',
            'cancel my account', 'close my account', 'done with',
            'fed up', 'unacceptable', 'ridiculous', 'three weeks',
            'multiple times', 'keep telling', 'nobody can help'
        ]
        
        text_lower = text.lower()
        return any(kw in text_lower for kw in escalation_keywords)
    
    def _detect_complaint_signal(self, text: str) -> bool:
        """
        Detect if text contains complaint signals.
        
        Args:
            text: Text to analyze.
            
        Returns:
            True if complaint signals detected.
        """
        complaint_keywords = [
            'complaint', 'disappointed', 'frustrated', 'terrible',
            'awful', 'worst', 'never again', 'poor service',
            'not satisfied', 'unhappy', 'problem', 'issue',
            'wrong', 'error', 'mistake', 'failed'
        ]
        
        text_lower = text.lower()
        return any(kw in text_lower for kw in complaint_keywords)
    
    def get_transcripts_by_outcome(self, outcome: str) -> List[Transcript]:
        """
        Filter transcripts by outcome.
        
        Args:
            outcome: Outcome to filter by.
            
        Returns:
            List of transcripts matching the outcome.
        """
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        return [t for t in self.transcripts if t.outcome == outcome]
    
    def get_turns_by_outcome(self, outcome: str) -> List[TurnDocument]:
        """
        Filter turn documents by outcome.
        
        Args:
            outcome: Outcome to filter by.
            
        Returns:
            List of turn documents matching the outcome.
        """
        return [t for t in self.turn_documents if t.outcome == outcome]
    
    def get_turns_by_transcript(self, transcript_id: str) -> List[TurnDocument]:
        """
        Get all turns for a specific transcript.
        
        Args:
            transcript_id: ID of the transcript.
            
        Returns:
            List of turn documents for the transcript.
        """
        return [t for t in self.turn_documents if t.transcript_id == transcript_id]
    
    def get_full_conversation(self, transcript_id: str) -> Optional[Transcript]:
        """
        Get the full transcript by ID.
        
        Args:
            transcript_id: ID of the transcript.
            
        Returns:
            Transcript object or None if not found.
        """
        for t in self.transcripts:
            if t.transcript_id == transcript_id:
                return t
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.
        
        Returns:
            Dictionary with data statistics.
        """
        if not self._loaded:
            return {"error": "Data not loaded"}
        
        # Count by outcome
        outcome_counts = {}
        for t in self.transcripts:
            outcome = t.outcome
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        # Count by domain
        domain_counts = {}
        for t in self.transcripts:
            domain = t.domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Average turns per conversation
        total_turns = sum(len(t.conversation) for t in self.transcripts)
        avg_turns = total_turns / len(self.transcripts) if self.transcripts else 0
        
        return {
            "total_transcripts": len(self.transcripts),
            "total_turns": len(self.turn_documents) if self.turn_documents else total_turns,
            "average_turns_per_conversation": round(avg_turns, 2),
            "outcomes": outcome_counts,
            "domains": domain_counts,
            "unique_intents": len(set(t.intent for t in self.transcripts))
        }
    
    def get_unique_outcomes(self) -> List[str]:
        """Get list of unique outcomes in the data."""
        if not self._loaded:
            return []
        return sorted(list(set(t.outcome for t in self.transcripts)))
    
    def get_unique_domains(self) -> List[str]:
        """Get list of unique domains in the data."""
        if not self._loaded:
            return []
        return sorted(list(set(t.domain for t in self.transcripts)))
    
    def export_turn_documents(self, output_path: str) -> None:
        """
        Export turn documents to JSON file.
        
        Args:
            output_path: Path to output file.
        """
        output = [td.to_dict() for td in self.turn_documents]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(output)} turn documents to {output_path}")


def create_enriched_text(turn: TurnDocument, include_context: bool = True) -> str:
    """
    Create enriched text for embedding that includes contextual information.
    
    Args:
        turn: TurnDocument to enrich.
        include_context: Whether to include domain/intent context.
        
    Returns:
        Enriched text string.
    """
    parts = []
    
    if include_context:
        parts.append(f"[{turn.domain}]")
        parts.append(f"[{turn.speaker}]")
    
    parts.append(turn.text)
    
    return " ".join(parts)


# Convenience function for quick loading
def load_and_process(file_path: Optional[str] = None) -> Tuple[DataProcessor, List[TurnDocument]]:
    """
    Convenience function to load and process data in one call.
    
    Args:
        file_path: Optional path to JSON file.
        
    Returns:
        Tuple of (DataProcessor, List[TurnDocument])
    """
    processor = DataProcessor()
    processor.load_data(file_path)
    turns = processor.flatten_to_turns()
    return processor, turns


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    processor.load_data()
    turns = processor.flatten_to_turns()
    
    print("\n=== Data Statistics ===")
    stats = processor.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Sample Turn Document ===")
    if turns:
        sample = turns[0]
        print(f"Transcript ID: {sample.transcript_id}")
        print(f"Speaker: {sample.speaker}")
        print(f"Text: {sample.text[:100]}...")
        print(f"Outcome: {sample.outcome}")
        print(f"Sentiment: {sample.sentiment_score}")
