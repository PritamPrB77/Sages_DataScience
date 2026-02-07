# 🔍 Causal RAG Analyzer

A research-grade, evidence-grounded causal analysis system for customer service conversations, built with **LangGraph** and **Streamlit**.

## 🎯 Overview

This system performs **Causal, Evidence-Constrained, Context-Preserving RAG** analysis to answer WHY specific outcomes (escalations, complaints, resolutions) occurred in customer service conversations.

### Key Principles

- ❌ **No Hallucinations**: All claims must be traceable to actual dialogue
- ❌ **No External Knowledge**: LLM reasons only over retrieved evidence
- ✅ **Evidence Grounded**: Every causal factor includes supporting quotes
- ✅ **Multi-Turn Context**: Follow-up questions maintain conversation state

## 🏗️ Architecture

```
JSON File (.json)
   ↓
Conversation Parser
   ↓
Turn-Level Chunking
   ↓
Embedding Model (Sentence-Transformers)
   ↓
Vector Store (FAISS)
   ↓
LangGraph RAG Pipeline
   ├── Query Understanding Node
   ├── RAG Retrieval Node
   ├── Causal Reasoning Node
   ├── Evidence Validation Node
   └── Context Memory Node
   ↓
Structured Output
   ↓
Streamlit UI
```

## 📁 Project Structure

```
Pravaah2/
├── app.py                      # Streamlit UI application
├── config.py                   # Configuration and constants
├── data_processor.py           # JSON parsing and turn-level chunking
├── embedding_store.py          # FAISS vector store and embeddings
├── context_memory.py           # Multi-turn context management
├── causal_reasoning.py         # LLM-based causal analysis
├── langgraph_orchestrator.py   # LangGraph workflow orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── README.md                   # This file
└── Conversational_Transcript_Dataset.json  # Sample dataset
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Google Gemini API key
# Get your key from: https://makersuite.google.com/app/apikey
# GEMINI_API_KEY=your-key-here
```

Or set the environment variable directly:

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "your-key-here"

# Linux/macOS
export GEMINI_API_KEY="your-key-here"
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Task 1: Causal Explanation

Ask WHY a specific outcome occurred:

1. Load the dataset (click "Load Data" in sidebar)
2. Select an outcome (e.g., "Escalation")
3. Ask: "Why do calls escalate?"

The system will:
- Retrieve relevant dialogue turns
- Identify behavioral patterns
- Generate causal explanations with evidence

### Task 2: Follow-Up Reasoning

After initial analysis, ask follow-up questions:

- "Could this have been prevented?"
- "What agent behaviors contributed to this?"
- "Show me the evidence for factor 1"
- "Compare with non-escalated calls"

The system maintains context and reuses retrieved evidence.

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Embedding model
EmbeddingModel.MINILM  # sentence-transformers/all-MiniLM-L6-v2

# LLM settings (Gemini)
LLMConfig(
    model_name="gemini-1.5-flash",
    temperature=0.1,  # Low for deterministic reasoning
    max_tokens=2000
)

# RAG settings
RAGConfig(
    top_k_retrieval=15,
    similarity_threshold=0.5
)
```

## 📊 Dataset Format

The system expects JSON in this format:

```json
{
  "transcripts": [
    {
      "transcript_id": "unique-id",
      "time_of_interaction": "2025-01-01 10:00:00",
      "domain": "E-commerce & Retail",
      "intent": "Escalation - Repeated Service Failures",
      "reason_for_call": "Customer description...",
      "conversation": [
        {"speaker": "Agent", "text": "Hello, how can I help?"},
        {"speaker": "Customer", "text": "I've been trying..."}
      ]
    }
  ]
}
```

## 🔬 Technical Details

### LangGraph Workflow Nodes

1. **Query Understanding**: Detects intent (explain, compare, prevent, drill-down)
2. **RAG Retrieval**: Semantic search with metadata filtering
3. **Causal Reasoning**: LLM generates causal explanations
4. **Evidence Validation**: Blocks unsupported claims
5. **Context Memory**: Updates state for follow-up queries

### Evidence Validation

The system validates all LLM outputs:
- Checks that evidence quotes exist in retrieved context
- Adjusts confidence scores based on evidence support
- Removes factors without valid citations

### Supported Outcomes

- Escalation
- Complaint
- Refund
- Resolution
- Fraud Handled
- Service Recovery
- Customer Churn Risk
- Successful Resolution
- Unresolved

## 📝 API Reference

### CausalRAGOrchestrator

```python
from langgraph_orchestrator import CausalRAGOrchestrator

# Initialize
orchestrator = CausalRAGOrchestrator(
    data_path="path/to/data.json",
    force_rebuild=False
)

# Process query
result = orchestrator.process_query(
    query="Why do calls escalate?",
    outcome="Escalation",
    domain=None  # Optional filter
)

# Access results
print(result["response"])
print(result["causal_factors"])
print(result["evidence_display"])

# Reset context for new session
orchestrator.reset_context()

# Export analysis
orchestrator.export_analysis("output.json")
```

### Output Format

```json
{
  "outcome": "Escalation",
  "causal_factors": [
    {
      "factor_id": "cf_1",
      "name": "Repeated Unresolved Issues",
      "description": "Customer mentioned multiple attempts",
      "causal_explanation": "Repeated failures led to frustration...",
      "confidence": 0.85,
      "evidence_turn_ids": ["123-456:5", "123-456:7"],
      "evidence_snippets": ["I've been trying for three weeks..."]
    }
  ],
  "evidence": [...],
  "transcript_ids": ["123-456", "789-012"]
}
```

## 🧪 Testing

Run module tests:

```bash
# Test data processor
python data_processor.py

# Test embedding store
python embedding_store.py

# Test causal reasoning
python causal_reasoning.py

# Test full orchestrator
python langgraph_orchestrator.py
```

## ⚠️ Requirements

- Python 3.9+
- Google Gemini API key (for LLM reasoning) - Get it from https://makersuite.google.com/app/apikey
- ~500MB disk space for embeddings cache
- 8GB+ RAM recommended for large datasets

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please ensure:
- All causal claims remain evidence-grounded
- No hallucination-prone modifications
- Tests pass before submitting PR
