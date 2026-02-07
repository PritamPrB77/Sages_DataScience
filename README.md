# IITB - Dual Solution Repository for Conversational Causal Analysis

This repository contains **two implementations** of the same problem statement: explain conversational outcomes (for example escalation) with evidence, support follow-up reasoning, and provide actionable intervention insights.

## 1. Solution Comparison

### SOL1 (`SOL1/Pravaah2`)
- Approach: LangGraph-based RAG pipeline with LLM reasoning.
- Retrieval: Sentence-transformer embeddings + FAISS index.
- Reasoning: Google Gemini-backed causal explanation with evidence validation.
- Strength: Rich multi-stage orchestration and advanced context memory.
- Tradeoff: Requires API key (`GEMINI_API_KEY`) and online LLM access.

### SOL2 (`SOL2/Pravaah`)
- Approach: Deterministic preprocessing + interpretable signal mining + ML prediction + rule-based counterfactuals.
- Retrieval/Reasoning: Evidence index + intent-routed follow-up handler.
- UI: Multi-page Streamlit app.
- Strength: Modular, reproducible pipeline scripts and clear artifacts.
- Tradeoff: Simpler causal modeling than SOL1's graph + LLM pipeline.

## 2. How To Run

### Run SOL1
```powershell
cd SOL1\Pravaah2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GEMINI_API_KEY="your-api-key"
python -m streamlit run app.py
```

### Run SOL2
```powershell
cd SOL2\Pravaah
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app/Home.py
```

## 3. Complete Project Structure

Notes:
- The trees below include source, docs, data, artifacts, notebooks, and generated runtime folders currently present.
- `__pycache__` and `*.pyc` are auto-generated Python bytecode caches.

### Root Structure
```text
IITB/
|- README.md
|- SOL1/
|  |- README.md
|  |- Team_Sages_Hackathon_Submisssion.pdf
|  `- Pravaah2/
`- SOL2/
   |- README.md
   |- CAIR.pdf
   `- Pravaah/
```

### SOL1 Structure (`SOL1/Pravaah2`)
```text
SOL1/Pravaah2/
|- app.py
|- causal_reasoning.py
|- config.py
|- context_memory.py
|- data_processor.py
|- embedding_store.py
|- langgraph_orchestrator.py
|- requirements.txt
|- README.md
|- README_old.md
|- SETUP_INSTRUCTIONS.md
|- setup_api_key.ps1
|- Conversational_Transcript_Dataset.json
|- repaired_conversations.json
|- temp_upload.json
|- model_test.ipynb.ipynb
|- vector_index/
|  |- index.faiss
|  `- metadata.pkl
`- __pycache__/
   |- causal_reasoning.cpython-313.pyc
   |- config.cpython-313.pyc
   |- context_memory.cpython-313.pyc
   |- data_processor.cpython-313.pyc
   |- embedding_store.cpython-313.pyc
   `- langgraph_orchestrator.cpython-313.pyc
```

### SOL2 Structure (`SOL2/Pravaah`)
```text
SOL2/Pravaah/
|- README.md
|- LICENSE
|- requirements.txt
|- preprocessing/
|  |- __init__.py
|  |- ingest_validate.py
|  |- normalize.py
|  `- __pycache__/ingest_validate.cpython-313.pyc
|- features/
|  |- __init__.py
|  |- __init__.pytouch
|  |- turn_features.py
|  `- conversation_features.py
|- casual_analysis/
|  |- __init__.py
|  |- signal_miner.py
|  `- casual_scoring.py
|- evidence_index/
|  |- __init__.py
|  |- __init__.pytouch
|  `- build_index.py
|- ml_pipeline/
|  |- __init__.py
|  |- build_single_feature.py
|  |- predict_outcome.py
|  |- rebuild_label_mapping.py
|  |- test_load_model.py
|  |- test_predict.py
|  |- test_real_prediction.py
|  |- tempCodeRunnerFile.py
|  `- __pycache__/
|     |- __init__.cpython-313.pyc
|     |- build_single_feature.cpython-313.pyc
|     `- predict_outcome.cpython-313.pyc
|- query_engine/
|  |- __init__.py
|  |- __init__.pytouch
|  |- intent_classifier.py
|  |- followup_handler.py
|  |- task1_explainer.py
|  `- __pycache__/
|     |- __init__.cpython-313.pyc
|     |- followup_handler.cpython-313.pyc
|     |- intent_classifier.cpython-313.pyc
|     `- task1_explainer.cpython-313.pyc
|- context_memory/
|  |- __init__.py
|  |- __init__.pytouch
|  |- analysis_context.py
|  |- memory_store.py
|  `- __pycache__/
|     |- __init__.cpython-313.pyc
|     `- analysis_context.cpython-313.pyc
|- counterfactual/
|  |- __init__.py
|  |- counterfactual_rules.py
|  |- counterfactual_engine.py
|  `- __pycache__/
|     |- __init__.cpython-313.pyc
|     |- counterfactual_engine.cpython-313.pyc
|     `- counterfactual_rules.cpython-313.pyc
|- streamlit_app/
|  |- Home.py
|  `- pages/
|     |- 1_Overview.py
|     |- 2_Conversation_Explorer.py
|     |- 3_Causal_Analysis.py
|     |- 4_Multi_Turn_Query.py
|     `- 5_Counterfactual_Simulator.py
|- data/
|  |- schema.md
|  |- raw/Conversational_Transcript_Dataset.json
|  `- processed/
|     |- conversations.json
|     |- conversations_with_features.json
|     `- conversations_with_conv_features.json
|- artifacts/
|  |- candidate_signals.json
|  |- scored_causal_signals.json
|  |- evidence_index.json
|  |- label_mapping.json
|  `- outcome_predictor.keras
|- EDA/
|  |- eda_summary_report.txt
|  |- clean_conversational_dataset.csv
|  |- clean_conversational_dataset.json
|  |- clean_conversational_dataset.jsonl
|  |- clean_train_dataset.json
|  |- clean_validation_dataset.json
|  `- clean_test_dataset.json
`- notebooks/
   |- CAIR_CD_EDA.ipynb
   |- CAIR_CD_CLEANING.ipynb
   |- CAIR_CD_MODEL_TRAINING.ipynb
   `- outputs/
      |- label_mapping.json
      `- outcome_predictor.keras
```

## 4. File Descriptions and Functions

## SOL1 (`SOL1/Pravaah2`) - Source Modules

### `app.py`
- Purpose: Main Streamlit application.
- Key functions:
  - `init_session_state()`
  - `load_data(file_path, force_rebuild=False)`
  - `render_header()`
  - `render_sidebar()`
  - `render_evidence_card(evidence)`
  - `render_causal_factor(factor, index)`
  - `render_chat_message(message)`
  - `process_query(query)`
  - `render_main_interface()`
  - `render_statistics_dashboard()`
  - `main()`

### `config.py`
- Purpose: Central configuration and constants.
- Key classes/functions:
  - `EmbeddingModel`, `VectorStore` (Enums)
  - `EmbeddingConfig`, `VectorStoreConfig`, `LLMConfig`, `TransformerConfig`, `RAGConfig`, `CausalConfig`, `SystemConfig`
  - `get_config()`
  - `update_config(**kwargs)`

### `data_processor.py`
- Purpose: Parse transcript JSON into turn-level documents for retrieval.
- Key classes/functions:
  - `TurnDocument`, `Transcript`, `DataProcessor`
  - `create_enriched_text(turn, include_context=True)`
  - `load_and_process(file_path=None)`

### `embedding_store.py`
- Purpose: Embedding generation and FAISS retrieval store.
- Key classes/functions:
  - `SearchResult`, `EmbeddingModel`, `FAISSVectorStore`, `EmbeddingStore`
  - `create_embedding_store(...)`

### `context_memory.py`
- Purpose: Multi-turn context and intent tracking.
- Key classes/functions:
  - `QueryIntent` (Enum)
  - `CausalFactor`, `RetrievedEvidence`, `AnalysisResult`, `ConversationTurn`
  - `ContextMemory`
  - `detect_query_intent(query, has_prior_context=False)`

### `causal_reasoning.py`
- Purpose: Detect patterns, perform LLM reasoning, validate evidence.
- Key classes:
  - `PatternMatch`
  - `CausalPatternDetector`
  - `LLMReasoner`
  - `EvidenceValidator`
  - `CausalReasoningEngine`

### `langgraph_orchestrator.py`
- Purpose: End-to-end LangGraph workflow orchestration.
- Key classes/functions:
  - `GraphState`
  - `QueryUnderstandingNode`
  - `RAGRetrievalNode`
  - `CausalReasoningNode`
  - `EvidenceValidationNode`
  - `ContextMemoryNode`
  - `CausalRAGOrchestrator`
  - `create_orchestrator(...)`

## SOL1 - Supporting Files
- `requirements.txt`: dependency list.
- `setup_api_key.ps1`: interactive Gemini API key setup script.
- `SETUP_INSTRUCTIONS.md`: setup and troubleshooting guide.
- `README.md`, `README_old.md`: documentation versions.
- `Conversational_Transcript_Dataset.json`: main input dataset.
- `repaired_conversations.json`, `temp_upload.json`: additional data variants.
- `vector_index/*`: persisted FAISS index and metadata.
- `model_test.ipynb.ipynb`: notebook experimentation.

## SOL2 (`SOL2/Pravaah`) - Source Modules

### Preprocessing
#### `preprocessing/ingest_validate.py`
- Purpose: Load and validate raw dataset format and speaker/text constraints.
- Key functions:
  - `load_raw_dataset(path=RAW_DATA_PATH)`
  - `validate_dataset(conversations)`

#### `preprocessing/normalize.py`
- Purpose: Normalize schema and derive `outcome_event` labels.
- Key functions:
  - `map_outcome_event(intent)`
  - `normalize_conversations(raw_conversations)`

### Features
#### `features/turn_features.py`
- Purpose: Turn-level feature extraction.
- Key function:
  - `extract_turn_features(conversations)`

#### `features/conversation_features.py`
- Purpose: Conversation-level aggregate features.
- Key function:
  - `add_conversation_features(conversations)`

### Causal Signal Pipeline
#### `casual_analysis/signal_miner.py`
- Purpose: Mine candidate interpretable signals from conversation features.
- Key function:
  - `mine_candidate_signals(conversations)`

#### `casual_analysis/casual_scoring.py`
- Purpose: Score candidate signals using outcome-conditioned lift.
- Key functions:
  - `load_total_outcomes(conversations)`
  - `score_signals(candidate_signals, outcome_counts, total_outcomes)`

#### `evidence_index/build_index.py`
- Purpose: Build traceable evidence index from strong causal signals.
- Key functions:
  - `identify_turn_span(convo, signal)`
  - `build_evidence_index(conversations, scored_signals)`

### ML
#### `ml_pipeline/build_single_feature.py`
- Purpose: Build feature vector for inference (deterministic + embedding).
- Key functions:
  - `build_conversation_text(convo)`
  - `build_feature_vector(conversation)`

#### `ml_pipeline/predict_outcome.py`
- Purpose: Load trained Keras model and return predicted label + confidence.
- Key function:
  - `predict_outcome(conversation)`

#### `ml_pipeline/rebuild_label_mapping.py`
- Purpose: Rebuild `artifacts/label_mapping.json` from training labels.

#### `ml_pipeline/test_load_model.py`, `ml_pipeline/test_predict.py`, `ml_pipeline/test_real_prediction.py`
- Purpose: Model loading and inference smoke tests.

### Query Engine and Context
#### `query_engine/intent_classifier.py`
- Purpose: Semantic intent detection for follow-up queries.
- Key function:
  - `predict_intent(query, threshold=0.45)`

#### `query_engine/followup_handler.py`
- Purpose: Intent-based follow-up response generation.
- Key function:
  - `handle_followup(query, context)`

#### `query_engine/task1_explainer.py`
- Purpose: Main router for fresh explanation, follow-up, and counterfactual requests.
- Key functions:
  - `parse_outcome_from_query(query)`
  - `explain_outcome(query, max_evidence=5)`
  - `answer_query(query)`

#### `context_memory/analysis_context.py`
- Purpose: Stores most recent query/result context.
- Key class:
  - `AnalysisContext`

#### `context_memory/memory_store.py`
- Purpose: Alternative context memory object with signal/history tracking.
- Key class:
  - `ContextMemory`

### Counterfactual
#### `counterfactual/counterfactual_rules.py`
- Purpose: Rule base mapping factors to interventions.

#### `counterfactual/counterfactual_engine.py`
- Purpose: Convert top causal factors into preventive interventions.
- Key function:
  - `generate_counterfactuals(context, top_k=2)`

### Streamlit Application
#### `streamlit_app/Home.py`
- Purpose: Main dashboard integrating causal analysis, follow-up, and counterfactual simulation.

#### `streamlit_app/pages/1_Overview.py`
- Purpose: High-level project overview and architecture summary.

#### `streamlit_app/pages/2_Conversation_Explorer.py`
- Purpose: Browse conversations and highlight evidence spans.

#### `streamlit_app/pages/3_Causal_Analysis.py`
- Purpose: Standalone causal analysis and ML prediction page.

#### `streamlit_app/pages/4_Multi_Turn_Query.py`
- Purpose: Chat-style multi-turn analytical interface.
- Key helper:
  - `json_to_text(resp)`

#### `streamlit_app/pages/5_Counterfactual_Simulator.py`
- Purpose: Standalone counterfactual intervention simulator.

## SOL2 - Data/Artifacts/Docs
- `data/schema.md`: canonical schema specification.
- `data/raw/*`, `data/processed/*`: raw and transformed datasets.
- `artifacts/*`: mined signals, evidence index, label mapping, trained model.
- `EDA/*`: cleaned datasets and EDA report.
- `notebooks/*`: EDA/cleaning/training notebooks and notebook output artifacts.
- `LICENSE`: GPLv3 license.

## 5. SOL2 Pipeline Execution Order

Run from `SOL2/Pravaah`:

```powershell
python preprocessing/ingest_validate.py
python preprocessing/normalize.py
python features/turn_features.py
python features/conversation_features.py
python casual_analysis/signal_miner.py
python casual_analysis/casual_scoring.py
python evidence_index/build_index.py
```

## 6. Practical Notes
- SOL1 needs `GEMINI_API_KEY` and network access for LLM reasoning.
- SOL2 is mostly local but first-time transformer usage downloads model weights.
- Files like `*.pytouch`, `tempCodeRunnerFile.py`, and `__pycache__` are auxiliary/generated and can be excluded in production cleanup.