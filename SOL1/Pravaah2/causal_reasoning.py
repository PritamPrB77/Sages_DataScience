"""
Causal Reasoning Module for Causal RAG System.
Implements causal factor identification, explanation generation, and evidence validation.
Uses LLM for reasoning while ensuring all claims are grounded in retrieved evidence.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import os

from config import (
    get_config, 
    CAUSAL_BEHAVIOR_PATTERNS,
    NEGATIVE_SENTIMENT_KEYWORDS,
    AGENT_POSITIVE_BEHAVIORS,
    AGENT_NEGATIVE_BEHAVIORS
)
from data_processor import TurnDocument
from embedding_store import SearchResult
from context_memory import CausalFactor, RetrievedEvidence, AnalysisResult, QueryIntent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Represents a detected behavioral pattern."""
    pattern_name: str
    pattern_type: str  # e.g., "escalation_signal", "agent_behavior"
    confidence: float
    evidence_turns: List[Tuple[str, int, str]]  # (transcript_id, turn_id, text)


class CausalPatternDetector:
    """
    Detects causal behavior patterns in conversation turns.
    Uses rule-based pattern matching as a first pass before LLM reasoning.
    """
    
    def __init__(self):
        """Initialize the pattern detector."""
        self.escalation_patterns = {
            "repeated_unresolved_issues": [
                r"(three|3|multiple|several)\s*(week|time|call|attempt)",
                r"(tried|trying|been|keep)\s+(to\s+)?(resolve|fix|get|contact)",
                r"(no one|nobody|nothing)\s+(can|has|is)\s+(help|fixed|resolved)",
                r"(still|again|yet another)\s+(not|hasn't|haven't)"
            ],
            "agent_dismissiveness": [
                r"(policy|cannot|can't|won't|unable)\s+(allow|do|help|change)",
                r"there('s| is)\s+nothing\s+(I|we)\s+can",
                r"(that's|this is)\s+(not|how)\s+(possible|works)"
            ],
            "transfer_fatigue": [
                r"(transferred|transfer|passed)\s+(me\s+)?(to|around|multiple)",
                r"(different|another|new)\s+(person|agent|representative)",
                r"explain(ed|ing)?\s+(again|multiple|every)"
            ],
            "escalation_request": [
                r"(speak|talk)\s+(to|with)\s+(supervisor|manager|someone\s+else)",
                r"(want|need|demand)\s+(a\s+)?(supervisor|manager)",
                r"escalat(e|ion)",
                r"(formal\s+)?complaint"
            ],
            "customer_frustration": [
                r"(frustrated|furious|angry|upset|disappointed)",
                r"(unacceptable|ridiculous|terrible|awful)",
                r"(fed\s+up|had\s+enough|done\s+with)",
                r"(waste|wasted)\s+(of\s+)?(my\s+)?time"
            ],
            "unfulfilled_promises": [
                r"(told|said|promised)\s+(me\s+)?(it\s+)?(would|will|should)",
                r"(supposed\s+to|should\s+have)\s+(be|been|receive|get)",
                r"never\s+(received|got|happened|came)"
            ]
        }
        
        self.agent_behavior_patterns = {
            "empathy_shown": [
                r"(sorry|apologize|understand|appreciate)",
                r"(frustrated|difficult|inconvenient)",
                r"(help\s+you|assist\s+you|take\s+care)"
            ],
            "solution_offered": [
                r"(let\s+me|I\s+can|I'll|we\s+can)",
                r"(fix|resolve|solve|address)\s+(this|that|your)",
                r"(refund|replacement|credit|compensation)"
            ],
            "ownership_taken": [
                r"I('ll|\s+will)\s+(personally|make\s+sure|ensure)",
                r"(my\s+responsibility|take\s+ownership)",
                r"I('ll|\s+will)\s+follow\s+up"
            ],
            "defensive_response": [
                r"(policy|procedure)\s+(states|requires|says)",
                r"there('s| is)\s+nothing\s+(more\s+)?(I|we)\s+can",
                r"(not\s+my|not\s+our)\s+(fault|responsibility|department)"
            ]
        }
    
    def detect_patterns(
        self, 
        turns: List[SearchResult], 
        outcome: str
    ) -> List[PatternMatch]:
        """
        Detect behavioral patterns in retrieved turns.
        
        Args:
            turns: List of SearchResult objects.
            outcome: The outcome being analyzed.
            
        Returns:
            List of detected PatternMatch objects.
        """
        matches = []
        
        # Get relevant patterns for this outcome
        outcome_patterns = CAUSAL_BEHAVIOR_PATTERNS.get(outcome, [])
        
        for result in turns:
            doc = result.document
            text_lower = doc.text.lower()
            
            # Check escalation patterns (for customer turns)
            if doc.speaker == "Customer":
                for pattern_name, regexes in self.escalation_patterns.items():
                    for regex in regexes:
                        if re.search(regex, text_lower):
                            match = PatternMatch(
                                pattern_name=pattern_name,
                                pattern_type="customer_behavior",
                                confidence=result.score,
                                evidence_turns=[(doc.transcript_id, doc.turn_id, doc.text)]
                            )
                            matches.append(match)
                            break  # One match per pattern per turn
            
            # Check agent behavior patterns
            if doc.speaker == "Agent":
                for pattern_name, regexes in self.agent_behavior_patterns.items():
                    for regex in regexes:
                        if re.search(regex, text_lower):
                            match = PatternMatch(
                                pattern_name=pattern_name,
                                pattern_type="agent_behavior",
                                confidence=result.score,
                                evidence_turns=[(doc.transcript_id, doc.turn_id, doc.text)]
                            )
                            matches.append(match)
                            break
        
        return matches
    
    def aggregate_patterns(self, matches: List[PatternMatch]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate pattern matches to identify recurring themes.
        
        Args:
            matches: List of PatternMatch objects.
            
        Returns:
            Dictionary mapping pattern names to aggregated data.
        """
        aggregated = {}
        
        for match in matches:
            name = match.pattern_name
            if name not in aggregated:
                aggregated[name] = {
                    "count": 0,
                    "pattern_type": match.pattern_type,
                    "total_confidence": 0.0,
                    "evidence": []
                }
            
            aggregated[name]["count"] += 1
            aggregated[name]["total_confidence"] += match.confidence
            aggregated[name]["evidence"].extend(match.evidence_turns)
        
        # Calculate average confidence
        for name, data in aggregated.items():
            data["avg_confidence"] = data["total_confidence"] / data["count"]
        
        return aggregated


class LLMReasoner:
    """
    Uses LLM for causal reasoning over retrieved evidence.
    Ensures all generated explanations are grounded in provided context.
    Uses Google Gemini API for inference.
    """
    
    def __init__(self, config=None):
        """Initialize the LLM reasoner."""
        self.config = config or get_config()
        self._model = None
        
    def _get_model(self):
        """Get or create Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai
                api_key = self.config.llm.api_key or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
                genai.configure(api_key=api_key)
                
                # Configure generation settings
                generation_config = genai.GenerationConfig(
                    temperature=self.config.llm.temperature,
                    max_output_tokens=self.config.llm.max_tokens,
                )
                
                self._model = genai.GenerativeModel(
                    model_name=self.config.llm.model_name,
                    generation_config=generation_config
                )
            except ImportError:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        return self._model
    
    def generate_causal_explanation(
        self,
        outcome: str,
        evidence: List[SearchResult],
        patterns: Dict[str, Dict[str, Any]],
        query_intent: QueryIntent,
        conversation_context: str = ""
    ) -> Dict[str, Any]:
        """
        Generate causal explanation using Gemini LLM.
        
        Args:
            outcome: The outcome being explained.
            evidence: Retrieved evidence turns.
            patterns: Detected behavioral patterns.
            query_intent: User's query intent.
            conversation_context: Prior conversation summary.
            
        Returns:
            Dictionary with causal factors and explanations.
        """
        model = self._get_model()
        
        # Build evidence context
        evidence_text = self._format_evidence(evidence)
        patterns_text = self._format_patterns(patterns)
        
        prompt = f"""You are a causal reasoning expert analyzing customer service conversations.
Your task is to identify WHY specific outcomes (like escalations, complaints, resolutions) occurred.

CRITICAL RULES:
1. ONLY use information from the provided evidence. Never make up facts.
2. Every causal factor MUST be supported by specific quotes from the evidence.
3. Explain the CAUSAL CHAIN - how one thing led to another.
4. Distinguish between correlation and causation.
5. Rate your confidence for each factor (0.0 to 1.0).
6. If evidence is insufficient, say so rather than speculating.

Analyze why the outcome "{outcome}" occurred based on this evidence:

DETECTED BEHAVIORAL PATTERNS:
{patterns_text}

CONVERSATION EVIDENCE:
{evidence_text}

{f"PRIOR CONTEXT: {conversation_context}" if conversation_context else ""}

Identify the causal factors that led to this {outcome}. Focus on:
1. What specific behaviors or events triggered the outcome
2. How one thing caused or led to another
3. Which factors were most impactful

You MUST respond with ONLY valid JSON in this exact format (no markdown, no extra text):
{{
    "causal_factors": [
        {{
            "name": "Short descriptive name",
            "description": "What happened",
            "causal_explanation": "WHY this led to the outcome (the causal chain)",
            "confidence": 0.85,
            "evidence_quotes": ["exact quote 1", "exact quote 2"],
            "evidence_refs": ["transcript_id:turn_id"]
        }}
    ],
    "summary": "Brief summary of why the outcome occurred",
    "key_insights": ["insight 1", "insight 2"]
}}"""

        try:
            response = model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Handle potential markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            return result
            
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            # Return a fallback based on detected patterns
            return self._fallback_reasoning(outcome, patterns)
    
    def generate_followup_response(
        self,
        query: str,
        query_intent: QueryIntent,
        context: Dict[str, Any],
        evidence: List[RetrievedEvidence]
    ) -> str:
        """
        Generate response for follow-up queries using Gemini.
        
        Args:
            query: User's follow-up question.
            query_intent: Detected intent.
            context: Current context state.
            evidence: Retrieved evidence.
            
        Returns:
            Response string.
        """
        model = self._get_model()
        
        # Build context
        evidence_text = "\n".join([
            f"[{e.transcript_id}:{e.turn_id}] {e.speaker}: {e.text}"
            for e in evidence[:20]  # Limit for token efficiency
        ])
        
        factors_text = ""
        if context.get("causal_factors"):
            factors_text = "Previously identified causal factors:\n"
            for cf in context["causal_factors"]:
                factors_text += f"- {cf['name']}: {cf['causal_explanation']}\n"
        
        intent_guidance = {
            QueryIntent.PREVENTION: "Focus on what could have prevented the outcome.",
            QueryIntent.AGENT_ANALYSIS: "Focus on agent behaviors and their impact.",
            QueryIntent.CUSTOMER_ANALYSIS: "Focus on customer behaviors and emotions.",
            QueryIntent.DRILL_DOWN: "Provide more detail on the previously identified factors.",
            QueryIntent.SHOW_EVIDENCE: "Quote specific evidence that supports the analysis.",
            QueryIntent.COMPARE_OUTCOMES: "Compare with contrasting cases if available."
        }
        
        guidance = intent_guidance.get(query_intent, "Provide a helpful response.")

        prompt = f"""You are a causal analysis assistant. Answer follow-up questions about conversational outcomes.

RULES:
1. Only use information from the provided evidence and prior analysis.
2. Cite specific evidence when making claims.
3. If you don't have enough information, say so.
4. Keep responses focused and concise.

Context:
Analyzing outcome: {context.get('active_outcome', 'Unknown')}

{factors_text}

Evidence:
{evidence_text}

User question: {query}

Guidance: {guidance}

Provide a response that directly addresses the question using the available evidence."""

        try:
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return f"I encountered an error processing your question. Please try rephrasing."
    
    def _format_evidence(self, evidence: List[SearchResult]) -> str:
        """Format evidence for LLM prompt."""
        lines = []
        for i, result in enumerate(evidence[:30], 1):  # Limit for token efficiency
            doc = result.document
            lines.append(
                f"{i}. [{doc.transcript_id}:{doc.turn_id}] {doc.speaker}: \"{doc.text}\""
            )
        return "\n".join(lines)
    
    def _format_patterns(self, patterns: Dict[str, Dict[str, Any]]) -> str:
        """Format detected patterns for LLM prompt."""
        if not patterns:
            return "No clear patterns detected."
        
        lines = []
        for name, data in sorted(patterns.items(), key=lambda x: -x[1]["count"]):
            lines.append(
                f"- {name}: occurred {data['count']} times "
                f"(avg confidence: {data['avg_confidence']:.2f})"
            )
        return "\n".join(lines)
    
    def _fallback_reasoning(self, outcome: str, patterns: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate fallback reasoning when LLM fails."""
        factors = []
        
        for name, data in sorted(patterns.items(), key=lambda x: -x[1]["count"])[:5]:
            evidence_refs = [f"{t[0]}:{t[1]}" for t in data["evidence"][:3]]
            evidence_quotes = [t[2][:100] for t in data["evidence"][:3]]
            
            factors.append({
                "name": name.replace("_", " ").title(),
                "description": f"Pattern '{name}' was detected {data['count']} times",
                "causal_explanation": f"This behavioral pattern likely contributed to the {outcome}.",
                "confidence": min(data["avg_confidence"], 0.7),
                "evidence_quotes": evidence_quotes,
                "evidence_refs": evidence_refs
            })
        
        return {
            "causal_factors": factors,
            "summary": f"Analysis based on detected behavioral patterns for {outcome}.",
            "key_insights": [f"{len(patterns)} distinct patterns were identified."]
        }


class EvidenceValidator:
    """
    Validates that all causal claims are supported by retrieved evidence.
    Blocks hallucinations by checking citations against actual evidence.
    """
    
    def __init__(self):
        """Initialize the validator."""
        pass
    
    def validate_explanation(
        self,
        explanation: Dict[str, Any],
        evidence: List[SearchResult]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate that explanation is grounded in evidence.
        
        Args:
            explanation: Generated explanation from LLM.
            evidence: Retrieved evidence.
            
        Returns:
            Tuple of (validated explanation, list of warnings).
        """
        warnings = []
        validated = explanation.copy()
        validated["causal_factors"] = []
        
        # Build evidence lookup
        evidence_lookup = {}
        evidence_texts = []
        for result in evidence:
            doc = result.document
            key = f"{doc.transcript_id}:{doc.turn_id}"
            evidence_lookup[key] = doc.text
            evidence_texts.append(doc.text.lower())
        
        # Validate each causal factor
        for factor in explanation.get("causal_factors", []):
            validated_factor = factor.copy()
            valid_quotes = []
            valid_refs = []
            
            # Check each evidence reference
            for ref in factor.get("evidence_refs", []):
                if ref in evidence_lookup:
                    valid_refs.append(ref)
            
            # Check each evidence quote
            for quote in factor.get("evidence_quotes", []):
                quote_lower = quote.lower()
                # Check if quote appears in any evidence
                if any(quote_lower in text for text in evidence_texts):
                    valid_quotes.append(quote)
                else:
                    # Try fuzzy match - check if most words appear
                    quote_words = set(quote_lower.split())
                    for text in evidence_texts:
                        text_words = set(text.split())
                        overlap = len(quote_words & text_words) / max(len(quote_words), 1)
                        if overlap > 0.7:
                            valid_quotes.append(quote)
                            break
            
            # Only include factors with valid evidence
            if valid_quotes or valid_refs:
                validated_factor["evidence_quotes"] = valid_quotes
                validated_factor["evidence_refs"] = valid_refs
                
                # Adjust confidence based on evidence support
                original_count = len(factor.get("evidence_quotes", [])) + len(factor.get("evidence_refs", []))
                valid_count = len(valid_quotes) + len(valid_refs)
                
                if original_count > 0:
                    evidence_ratio = valid_count / original_count
                    validated_factor["confidence"] *= evidence_ratio
                
                validated["causal_factors"].append(validated_factor)
            else:
                warnings.append(
                    f"Factor '{factor.get('name', 'Unknown')}' removed: no valid evidence support"
                )
        
        if warnings:
            logger.warning(f"Validation warnings: {warnings}")
        
        return validated, warnings


class CausalReasoningEngine:
    """
    Main causal reasoning engine that orchestrates pattern detection,
    LLM reasoning, and evidence validation.
    """
    
    def __init__(self, config=None):
        """Initialize the reasoning engine."""
        self.config = config or get_config()
        self.pattern_detector = CausalPatternDetector()
        self.llm_reasoner = LLMReasoner(config)
        self.evidence_validator = EvidenceValidator()
    
    def analyze(
        self,
        outcome: str,
        evidence: List[SearchResult],
        query_intent: QueryIntent = QueryIntent.EXPLAIN_OUTCOME,
        conversation_context: str = ""
    ) -> AnalysisResult:
        """
        Perform full causal analysis.
        
        Args:
            outcome: Outcome to explain.
            evidence: Retrieved evidence.
            query_intent: User's query intent.
            conversation_context: Prior conversation summary.
            
        Returns:
            AnalysisResult object.
        """
        logger.info(f"Starting causal analysis for outcome: {outcome}")
        
        # Step 1: Detect behavioral patterns
        patterns = self.pattern_detector.detect_patterns(evidence, outcome)
        aggregated_patterns = self.pattern_detector.aggregate_patterns(patterns)
        
        logger.info(f"Detected {len(aggregated_patterns)} unique patterns")
        
        # Step 2: Generate causal explanation with LLM
        explanation = self.llm_reasoner.generate_causal_explanation(
            outcome=outcome,
            evidence=evidence,
            patterns=aggregated_patterns,
            query_intent=query_intent,
            conversation_context=conversation_context
        )
        
        # Step 3: Validate explanation against evidence
        validated_explanation, warnings = self.evidence_validator.validate_explanation(
            explanation, evidence
        )
        
        # Step 4: Build AnalysisResult
        causal_factors = []
        for factor_data in validated_explanation.get("causal_factors", []):
            factor = CausalFactor(
                factor_id=f"cf_{len(causal_factors) + 1}",
                name=factor_data.get("name", "Unknown"),
                description=factor_data.get("description", ""),
                causal_explanation=factor_data.get("causal_explanation", ""),
                confidence=factor_data.get("confidence", 0.5),
                evidence_turn_ids=factor_data.get("evidence_refs", []),
                evidence_snippets=factor_data.get("evidence_quotes", [])
            )
            causal_factors.append(factor)
        
        # Convert search results to retrieved evidence
        retrieved_evidence = [
            RetrievedEvidence(
                transcript_id=r.document.transcript_id,
                turn_id=r.document.turn_id,
                speaker=r.document.speaker,
                text=r.document.text,
                relevance_score=r.score
            )
            for r in evidence
        ]
        
        # Get unique transcript IDs
        transcript_ids = list(set(r.document.transcript_id for r in evidence))
        
        result = AnalysisResult(
            outcome=outcome,
            causal_factors=causal_factors,
            evidence=retrieved_evidence,
            transcript_ids=transcript_ids
        )
        
        logger.info(f"Analysis complete: {len(causal_factors)} causal factors identified")
        
        return result
    
    def answer_followup(
        self,
        query: str,
        query_intent: QueryIntent,
        context: Dict[str, Any],
        evidence: List[RetrievedEvidence]
    ) -> str:
        """
        Answer a follow-up question.
        
        Args:
            query: User's question.
            query_intent: Detected intent.
            context: Current context state.
            evidence: Available evidence.
            
        Returns:
            Response string.
        """
        return self.llm_reasoner.generate_followup_response(
            query=query,
            query_intent=query_intent,
            context=context,
            evidence=evidence
        )
    
    def compute_factor_confidence(
        self,
        factor: CausalFactor,
        evidence_count: int,
        pattern_count: int
    ) -> float:
        """
        Compute confidence score for a causal factor.
        
        Args:
            factor: The causal factor.
            evidence_count: Number of evidence pieces.
            pattern_count: Number of pattern matches.
            
        Returns:
            Confidence score (0-1).
        """
        # Base confidence from LLM
        base = factor.confidence
        
        # Boost based on evidence support
        evidence_boost = min(evidence_count / 5, 0.2)  # Max 0.2 boost
        
        # Boost based on pattern frequency
        pattern_boost = min(pattern_count / 10, 0.15)  # Max 0.15 boost
        
        final = min(base + evidence_boost + pattern_boost, 1.0)
        
        return round(final, 3)


if __name__ == "__main__":
    # Test the causal reasoning engine
    from data_processor import load_and_process
    from embedding_store import create_embedding_store
    
    print("Loading data...")
    processor, turns = load_and_process()
    
    # Use subset for testing
    test_turns = turns[:1000]
    
    print("Creating embedding store...")
    store = create_embedding_store(test_turns, force_rebuild=True)
    
    # Search for escalation-related turns
    print("\nSearching for escalation evidence...")
    results = store.vector_store.search(
        query="customer frustrated supervisor escalation unresolved multiple attempts",
        top_k=20,
        outcome_filter="Escalation"
    )
    
    if results:
        print(f"Found {len(results)} relevant turns")
        
        # Create reasoning engine and analyze
        engine = CausalReasoningEngine()
        
        print("\nRunning causal analysis...")
        analysis = engine.analyze(
            outcome="Escalation",
            evidence=results,
            query_intent=QueryIntent.EXPLAIN_OUTCOME
        )
        
        print("\n=== Analysis Results ===")
        print(f"Outcome: {analysis.outcome}")
        print(f"Transcripts analyzed: {len(analysis.transcript_ids)}")
        
        print("\n=== Causal Factors ===")
        for factor in analysis.causal_factors:
            print(f"\n{factor.name} (confidence: {factor.confidence:.2f})")
            print(f"  Description: {factor.description}")
            print(f"  Causal explanation: {factor.causal_explanation}")
            print(f"  Evidence count: {len(factor.evidence_snippets)}")
    else:
        print("No escalation evidence found in test data")
